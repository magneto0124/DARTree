"""N-gram trie builder for DARTree (migrated from DART's ngram_build.py).

Builds a ``.trie`` n-gram model (DART's binary format) from training text
using the C++ ``TrieNgram`` extension and parallel workers: each worker
processes a strided partition of the input items, saves a partial model
(``{order}gram-part{id}.trie``), and a final merge step folds the partials
together with ``add_all`` into ``{order}gram.trie``.

With ``--init-ngram PATH`` the build becomes an *update*: ``PATH`` points at
an existing DART ``.trie`` model, and the newly built data is merged ON TOP
of it (frequencies summed; the base model is counted exactly once) before
saving.  The base model's order must match ``--ngram-order``.

With ``--eval-output JSON`` the build additionally folds in the complete
input+output texts of a saved eval run (run_dartree.py output): inputs are
read from the dataset (replicating the eval run's shuffle/select via the
JSON's own ``summary.config``), outputs from the saved rows
(``dartree.text`` matched by ``sample_index`` / ``turn_index``), and every
sample's conversation text is added to the n-gram model.  No model inference
is run.

Data sources:
  * a directory of JSONL files, each line ``{"text": ...}`` (DART format), or
  * a DARTree dataset name understood by ``utils.data.load_and_process_dataset``
    (gsm8k, math500, aime25, alpaca, mt-bench, humaneval, mbpp, livecodebench);
    the dataset is detected automatically: existing directories are treated as
    JSONL, anything else as a dataset name,
  * plus (optionally) a saved eval output JSON via ``--eval-output``.

Usage::

    python utils/ngram_build.py --data /path/to/jsonl_dir \
        --output-path /path/to/out --ngram-order 3 --n-jobs 16

    # update an existing model instead of building from scratch
    python utils/ngram_build.py --data /path/to/more_jsonl \
        --output-path /path/to/out --ngram-order 3 --n-jobs 16 \
        --init-ngram /path/to/existing/3gram.trie

    # additionally fold saved eval input+output texts into the build
    python utils/ngram_build.py --data /path/to/jsonl_dir \
        --output-path /path/to/out --ngram-order 3 --n-jobs 16 \
        --eval-output results/gsm8k_pruned_t0.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from multiprocessing import Process
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm


DEFAULT_NGRAM_ORDER = 3
DEFAULT_N_JOBS = 16
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build an n-gram trie model (DART .trie format)."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "Directory of JSONL files (each line {\"text\": ...}) or a "
            "DARTree dataset name (gsm8k, math500, ...)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to save the partial and final n-gram models.",
    )
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=DEFAULT_NGRAM_ORDER,
        help=f"Order of the n-gram model. (default: {DEFAULT_NGRAM_ORDER})",
    )
    parser.add_argument(
        "--init-ngram",
        type=str,
        default=None,
        help=(
            "Optional path to an existing DART .trie model to UPDATE "
            "instead of building from scratch: the newly built data is "
            "merged on top of this model (frequencies summed, base counted "
            "once) and the result is saved as {order}gram.trie. The base "
            "model's order must match --ngram-order."
        ),
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help=(
            "Optional path to a saved eval output JSON (run_dartree.py "
            "output). When given, the complete input+output conversation "
            "texts of every sample are additionally read -- inputs from the "
            "dataset (order replicated from the JSON's summary.config), "
            "outputs from the saved rows -- and added to the n-gram model. "
            "No model inference is run."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel worker processes. (default: {DEFAULT_N_JOBS})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer name or path. (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--conversation-per-file",
        type=int,
        default=-1,
        help=(
            "JSONL mode: max conversations per file (-1 = all). Dataset "
            "mode: max conversations per worker partition (-1 = all). "
            "(default: -1)"
        ),
    )
    return parser.parse_args()


# ============================================================================
# Data Collection
# ============================================================================

def collect_items(data: str) -> List[str]:
    """Return the list of items to process.

    JSONL mode: absolute paths of every file under ``data`` (DART format).
    Dataset mode: a list of conversation texts (turns joined with newlines).
    """
    if os.path.isdir(data):
        items: List[str] = []
        for root, _, files in os.walk(data):
            for filename in files:
                items.append(os.path.join(root, filename))
        return items

    from utils.data import load_and_process_dataset

    dataset = load_and_process_dataset(data)
    texts: List[str] = []
    for row in dataset:
        turns = row.get("turns", [])
        texts.append("\n".join(str(t) for t in turns) if turns else "")
    return texts


# ============================================================================
# Eval-output collection (no inference)
# ============================================================================

def group_assistant_by_sample(
    rows: List[dict],
) -> Dict[int, Dict[int, str]]:
    """Map ``sample_index -> {turn_index: saved DARTree response text}``.

    Rows without a usable ``dartree.text`` are dropped.
    """
    out: Dict[int, Dict[int, str]] = {}
    for row in rows:
        resp = row.get("dartree") or {}
        text = resp.get("text")
        if not isinstance(text, str) or not text:
            continue
        sidx = int(row.get("sample_index", -1))
        tidx = int(row.get("turn_index", 0))
        out.setdefault(sidx, {})[tidx] = text
    return out


def build_conversation_text(
    user_turns: List[str],
    assistant_by_turn: Dict[int, str],
) -> str:
    """Join a sample's user turns with its saved assistant responses.

    Interleaves by turn index (u0, a0, u1, a1, ...); any assistant responses
    beyond the number of user turns are appended at the end.  Empty parts
    are dropped; the parts are joined with newlines, matching the
    conversation-text convention used by the dataset mode.
    """
    parts: List[str] = []
    num_turns = len(user_turns)
    for t in range(num_turns):
        if user_turns[t]:
            parts.append(user_turns[t])
        assistant = assistant_by_turn.get(t)
        if assistant:
            parts.append(assistant)
    for t in sorted(t for t in assistant_by_turn if t >= num_turns):
        if assistant_by_turn[t]:
            parts.append(assistant_by_turn[t])
    return "\n".join(parts)


def collect_eval_texts(args: argparse.Namespace) -> List[str]:
    """Full input+output conversation texts from a saved eval output JSON.

    Inputs come from the dataset (the eval run's shuffle/select is
    replicated from the JSON's own ``summary.config`` so ``sample_index``
    matches), outputs from the saved rows (``dartree.text`` matched by
    ``sample_index`` / ``turn_index``).  No model inference is run.
    """
    with open(args.eval_output, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    config = (data.get("summary") or {}).get("config") or {}

    dataset_name = config.get("dataset")
    if not dataset_name:
        raise ValueError(
            f"--eval-output {args.eval_output!r} has no summary.config.dataset; "
            "cannot reconstruct the eval inputs"
        )
    shuffle_seed = int(config.get("dataset_shuffle_seed", 0))
    max_samples = config.get("max_samples")

    from utils.data import load_and_process_dataset

    dataset = load_and_process_dataset(dataset_name)
    if max_samples is not None:
        count = max(0, int(max_samples))
        if len(dataset) > count:
            dataset = dataset.shuffle(seed=int(shuffle_seed))
        dataset = dataset.select(range(min(len(dataset), count)))

    assistant_by_sample = group_assistant_by_sample(rows)
    texts: List[str] = []
    missing = 0
    for sample_index, assistant_by_turn in sorted(
        assistant_by_sample.items()
    ):
        try:
            user_turns = [str(t) for t in dataset[sample_index]["turns"]]
        except (IndexError, KeyError) as exc:
            print(
                f"[warn] sample_index {sample_index} not found in dataset "
                f"(order mismatch?): {exc}"
            )
            missing += 1
            continue
        conversation = build_conversation_text(
            user_turns, assistant_by_turn
        )
        if conversation:
            texts.append(conversation)
    print(
        f"[eval-output] {args.eval_output}: {len(texts)} conversations "
        f"({missing} samples missing in dataset)"
    )
    return texts


# ============================================================================
# N-gram Building (Parallel Workers)
# ============================================================================

def build_ngram_partition(
    worker_id: int,
    args: argparse.Namespace,
    items: List[str],
) -> None:
    """Build the n-gram model for this worker's strided slice of items."""
    import transformers

    from utils.ngram_cpp import load_cpp_ngram

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.eos_token_id

    cpp_ngram = load_cpp_ngram()
    ngram_model = cpp_ngram.TrieNgram(order=args.ngram_order)

    item_indices = range(worker_id, len(items), args.n_jobs)
    is_primary_worker = worker_id == 0
    iterator = tqdm(item_indices, desc="Items") if is_primary_worker else item_indices

    for item_idx in iterator:
        item = items[item_idx]
        if os.path.isfile(item):
            _process_jsonl_file(
                filepath=item,
                tokenizer=tokenizer,
                ngram_model=ngram_model,
                eos_token_id=eos_token_id,
                conversation_limit=args.conversation_per_file,
            )
        else:
            _process_text(
                text=item,
                tokenizer=tokenizer,
                ngram_model=ngram_model,
                eos_token_id=eos_token_id,
            )

    output_file = os.path.join(
        args.output_path,
        f"{args.ngram_order}gram-part{worker_id}.trie",
    )
    ngram_model.save(output_file)


def _process_jsonl_file(
    filepath: str,
    tokenizer,
    ngram_model,
    eos_token_id: int,
    conversation_limit: int = -1,
) -> None:
    """Process a single JSONL file (each line ``{"text": ...}``)."""
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if conversation_limit != -1 and i >= conversation_limit:
                break
            data = json.loads(line)
            _process_text(
                text=data["text"],
                tokenizer=tokenizer,
                ngram_model=ngram_model,
                eos_token_id=eos_token_id,
            )


def _process_text(
    text: str,
    tokenizer,
    ngram_model,
    eos_token_id: int,
) -> None:
    """Tokenize a text, append EOS, and add it as one conversation."""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tokens.append(eos_token_id)
    ngram_model.add_conversation(tokens)


# ============================================================================
# N-gram Merging
# ============================================================================

def read_trie_order(path: str) -> int:
    """Read only the ``order`` header of a DART ``.trie`` file.

    The binary layout starts with ``[order: size_t][node_count: size_t]``
    (little-endian, 64-bit), so the order is available without loading the
    whole model -- used to fail fast when ``--init-ngram`` mismatches
    ``--ngram-order``.
    """
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(struct.calcsize("<Q")))[0]


def merge_partial_models(
    args: argparse.Namespace,
    partial_files: Sequence[str],
    init_path: Optional[str] = None,
) -> None:
    """Merge partial models into a single model, then clean up partials.

    Without ``init_path`` the first partial file seeds the merged model and
    the rest are folded in (build-from-scratch).  With ``init_path`` the
    existing model at that path is the base (counted exactly once) and EVERY
    partial file is new data folded on top of it (update mode).
    """
    if not partial_files:
        print("No partial files found to merge.")
        return

    from utils.ngram_cpp import load_cpp_ngram

    if init_path:
        print(f"Updating existing n-gram model: {init_path}")
    else:
        print(f"Merging {len(partial_files)} partial models...")
    cpp_ngram = load_cpp_ngram()
    if init_path:
        merged_model = cpp_ngram.TrieNgram.load(init_path)
        to_add = list(partial_files)
    else:
        merged_model = cpp_ngram.TrieNgram.load(partial_files[0])
        to_add = list(partial_files[1:])

    for partial_file in tqdm(to_add, desc="Merging"):
        partial_model = cpp_ngram.TrieNgram.load(partial_file)
        merged_model.add_all(partial_model)
        del partial_model  # Free memory

    final_output = os.path.join(args.output_path, f"{args.ngram_order}gram.trie")
    merged_model.save(final_output)
    print(f"Saved merged model to: {final_output}")

    for partial_file in partial_files:
        os.remove(partial_file)
    print(f"Removed {len(partial_files)} partial files.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Main entry point for n-gram model building."""
    args = parse_args()

    if args.init_ngram:
        init_order = read_trie_order(args.init_ngram)
        if init_order != args.ngram_order:
            raise ValueError(
                f"--init-ngram {args.init_ngram!r} has order {init_order} "
                f"but --ngram-order is {args.ngram_order}; they must match"
            )
        print(
            f"[ngram] will update existing model {args.init_ngram} "
            f"(order={init_order})"
        )

    os.makedirs(args.output_path, exist_ok=True)

    items = collect_items(args.data)
    if args.eval_output:
        # fold the saved eval input+output texts in as extra items; workers
        # treat non-file items as conversation texts (see build_ngram_partition)
        items = items + collect_eval_texts(args)
    if not items:
        raise RuntimeError(f"No training items found for --data {args.data!r}")
    print(f"Found {len(items)} training items.")

    processes = []
    for worker_id in range(args.n_jobs):
        process = Process(
            target=build_ngram_partition,
            args=(worker_id, args, items),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    partial_files = [
        os.path.join(args.output_path, f"{args.ngram_order}gram-part{i}.trie")
        for i in range(args.n_jobs)
    ]
    merge_partial_models(args, partial_files, init_path=args.init_ngram)

    print("Done!")


if __name__ == "__main__":
    main()
