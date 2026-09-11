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
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DARTree.")
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    # Public Domino checkpoint on Hugging Face.
    parser.add_argument("--draft-model", default="Huang2020/Qwen3-4B-Domino-b16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--variant", choices=["fixed", "pruned", "graft"], default="pruned")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-budget", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=64)
    parser.add_argument("--supertree-width", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--graft-ratio", type=float, default=0.6)
    parser.add_argument("--graft-k", type=int, default=8)
    parser.add_argument("--graft-template-depth", type=int)
    parser.add_argument("--graft-root-width", type=int, default=8)
    parser.add_argument("--graft-dedup", choices=["skip", "redirect"], default="skip")
    parser.add_argument("--graft-insert", choices=["root", "into_slot"], default="root")
    parser.add_argument("--graft-warmup", type=int, default=0)
    parser.add_argument("--graft-stages", type=str, default=None)
    parser.add_argument("--graft-no-prune", action="store_true")
    parser.add_argument("--graft-matrix-path", type=str, default=None)
    parser.add_argument("--graft-matrix-save", type=str, default=None)
    parser.add_argument("--record-round-trace", action="store_true")
    parser.add_argument(
        "--record-entropy", action="store_true",
        help=(
            "Print the TARGET model word-distribution entropy at each accepted "
            "draft-chain position (DARTree path and Domino chain baseline)."
        ),
    )
    args = parser.parse_args()

    from eval_dartree import main as evaluate

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
        "--output", output,
    ]
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
    if args.variant == "graft":
        engine_args += [
            "--graft-ratio", str(args.graft_ratio),
            "--graft-k", str(args.graft_k),
            "--graft-root-width", str(args.graft_root_width),
            "--graft-dedup", args.graft_dedup,
            "--graft-insert", args.graft_insert,
            "--graft-warmup", str(args.graft_warmup),
        ]
        if args.graft_template_depth is not None:
            engine_args += [
                "--graft-template-depth",
                str(args.graft_template_depth),
            ]
        if args.graft_stages is not None:
            engine_args += ["--graft-stages", args.graft_stages]
        if args.graft_no_prune:
            engine_args.append("--graft-no-prune")
        if args.graft_matrix_path is not None:
            engine_args += [
                "--graft-matrix-path", args.graft_matrix_path,
            ]
        if args.graft_matrix_save is not None:
            engine_args += [
                "--graft-matrix-save", args.graft_matrix_save,
            ]

    sys.argv = engine_args
    evaluate()


if __name__ == "__main__":
    main()
