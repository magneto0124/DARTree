# DARTree

This repository contains the official code release for **DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees**.

DARTree is a training-free speculative decoding method that extends causal correction from a single draft chain to multiple branches of a speculative tree. Starting from one block-parallel diffusion draft, DARTree performs depth-wise batched autoregressive correction to construct candidate branches efficiently, avoiding sequential correction-head inference during best-first tree expansion.

We provide two variants:

* **DARTree (fixed)** distributes a fixed verification budget across draft depths and directly verifies the resulting tree.
* **DARTree (pruned)** first constructs a wider fixed-width candidate supertree and then applies deferred top-B pruning to select the final verification tree.

This release includes the implementation used for the **Qwen3-4B** and **Qwen3-8B** experiments reported in the paper.

## Setup

```bash
python -m pip install -r requirements.txt
```

The code was tested with Python 3.12, PyTorch 2.8.0, Transformers 4.57.1, Triton 3.4.0, CUDA 12.8, and an NVIDIA RTX 6000 Ada GPU.

## Evaluation

The following command runs the main **DARTree (pruned)** configuration:

```bash
python run_dartree.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model Huang2020/Qwen3-4B-Domino-b16 \
  --dataset gsm8k \
  --variant pruned \
  --temperature 0
```

Use `--variant fixed` to evaluate **DARTree (fixed)**.

The default configuration uses a draft block size of 16, a verification budget of 64 nodes, 64 candidate tokens per position, a supertree width of 12 for the pruned variant, and `max_new_tokens=2048`. Default sample counts for each benchmark are specified in `run_dartree.py`.

Each evaluation produces a result file containing per-example generated token IDs, acceptance lengths, timing statistics, and an aggregate summary.
