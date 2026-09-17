from __future__ import annotations

import argparse
import sys
from pathlib import Path

# same with previous work
SAMPLE_COUNTS = {
    "gsm8k": 128,
    "math500": 128,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 128,
    "livecodebench": 128,
    "mt-bench": 80,
    "alpaca": 128,
    "sharegpt": 128,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DARTree.")
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    # Public Domino checkpoint on Hugging Face.
    parser.add_argument("--draft-model", default="Huang2020/Qwen3-4B-Domino-b16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--variant", choices=["fixed", "pruned"], default="pruned")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-budget", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--supertree-width", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ngram-model", default=None)
    parser.add_argument("--ngram-weight", type=float, default=0.0)
    parser.add_argument(
        "--nnt-lambda", type=float, default=None,
        help=(
            "NNT (next-next-token) correction mixture weight in [0, 1]; "
            "defaults to eval_dartree's NNT_MIX_LAMBDA. 1 disables the "
            "correction."
        ),
    )
    parser.add_argument(
        "--renorm-ngram", action="store_true",
        help=(
            "Renormalize each parent's ngram candidate probabilities over "
            "its candidate set before fusing with the draft term (see "
            "eval_dartree.py --renorm-ngram)."
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--record-round-trace", action="store_true")
    parser.add_argument(
        "--record-entropy", action="store_true",
        help=(
            "Print the TARGET model word-distribution entropy at each accepted "
            "draft-chain position (DARTree path and Domino chain baseline)."
        ),
    )
    parser.add_argument(
        "--record-rank-pairs", action="store_true",
        help=(
            "Record the (draft top-k rank, ngram top-k rank) of every accepted "
            "draft-chain token and write a CSV + first-quadrant scatter PNG "
            "(with the y=x diagonal) next to --output. "
            "Requires --ngram-weight > 0."
        ),
    )
    args = parser.parse_args()

    from eval_dartree import NNT_MIX_LAMBDA, main as evaluate

    nnt_lambda = (
        float(args.nnt_lambda)
        if args.nnt_lambda is not None
        else NNT_MIX_LAMBDA
    )

    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else SAMPLE_COUNTS.get(args.dataset, 128)
    )
    output = args.output or str(
        Path("results") / f"{args.dataset}_{args.variant}_t{args.temperature:g}.json"
    )

    engine_args = [
        "eval_dartree.py",
        "--target-model", args.target_model,
        "--draft-model", args.draft_model,
        "--dataset", args.dataset,
        "--max-samples", str(max_samples),
        "--dataset-shuffle-seed", "0",
        "--max-new-tokens", str(args.max_new_tokens),
        "--block-size", str(args.block_size),
        "--tree-budget", str(args.tree_budget),
        "--expansion-k", str(args.candidate_k),
        "--variant", args.variant,
        "--supertree-width", str(args.supertree_width),
        "--candidate-vocab-size", str(args.candidate_k),
        "--temperature", str(args.temperature),
        "--device", args.device,
        "--ngram-weight", str(args.ngram_weight),
        "--nnt-lambda", str(nnt_lambda),
        "--output", output,
    ]
    if args.ngram_model:
        engine_args += ["--ngram-model", args.ngram_model]
    if args.variant == "fixed":
        engine_args += [
            "--depth-bonus", "0",
            "--run-baselines",
        ]
    else:
        engine_args += [
            "--depth-bonus", "-0.2",
        ]
    if args.record_round_trace:
        engine_args.append("--record-round-trace")
    if args.record_entropy:
        engine_args.append("--record-entropy")
    if args.record_rank_pairs:
        engine_args.append("--record-rank-pairs")
    if args.renorm_ngram:
        engine_args.append("--renorm-ngram")

    sys.argv = engine_args
    evaluate()


if __name__ == "__main__":
    main()
