#!/usr/bin/env python3
"""Print, for the accepted draft chains during Domino speculative decoding, the
TARGET model's word-distribution entropy at each position of the chain.

Each ``[entropy]`` line is one accepted draft-chain position: the target model
distribution that accepted that draft token, measured as softmax(logits) entropy
on the same logits the sampler sees.

Usage examples
--------------
python run_entropy.py --target-model /local/Qwen3-4B \
    --draft-model /local/Qwen3-4B-Domino-b16 --device npu:0

python run_entropy.py --target-model Qwen/Qwen3-4B \
    --draft-model Huang2020/Qwen3-4B-Domino-b16 --device cuda:0 \
    --prompt "1+1=?"
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils import DFlashDraftModel, logits_entropy
from utils.device_backend import seed_all, set_device
from eval_dartree import normalize_draft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report target vs corrected-draft vocab-distribution entropy "
        "during Domino chain decoding."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", default="Huang2020/Qwen3-4B-Domino-b16")
    parser.add_argument("--prompt", default="Q: What is the capital of France?\nA:")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(0)
    device = torch.device(args.device)
    set_device(device)
    dtype = getattr(torch, args.dtype)

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model, attn_implementation="sdpa", dtype=dtype
        )
        .to(device)
        .eval()
    )
    draft_config = normalize_draft_config(
        AutoConfig.from_pretrained(args.draft_model)
    )
    draft = (
        DFlashDraftModel.from_pretrained(
            args.draft_model, config=draft_config,
            attn_implementation="sdpa", dtype=dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    print(
        f"\n=== Domino chain decode (block_size={args.block_size}, "
        f"temperature={args.temperature:g}, device={device}) ==="
    )
    print("Note: each line = one accepted draft-chain position, showing the TARGET\n"
          "model word-distribution entropy there; entropy = softmax of the logits\n"
          "used at sampling time (nats).\n")

    response = draft.spec_generate(
        input_ids=input_ids,
        target=target,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
        use_bias=True,
        record_entropy=True,
        return_dict=True,
    )

    print("\n=== Generated ===")
    generated = response.output_ids[0, response.num_input_tokens :]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"\n=== Entropy summary: {response.entropy}")


if __name__ == "__main__":
    main()
