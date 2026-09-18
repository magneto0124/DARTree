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

### Running on Huawei Ascend NPU

The DARTree *algorithm* is device-agnostic; only a few performance kernels were
NVIDIA-bound (CUDA graphs and inline NVIDIA Triton kernels). These are now routed
through `utils/device_backend.py` and are **disabled on Ascend**, where the code
uses equivalent pure-PyTorch eager fallbacks.

To run on Ascend:

1. Install the CANN toolkit and its matching `torch_npu` wheel (do **not** `pip
   install -r requirements.txt` for the CUDA `torch`/`triton` pins on this host —
   `torch_npu` bundles its own torch build). See the notes in
   `requirements.txt`.
2. Pass `--device npu:0` (or `npu:<idx>`):

   ```bash
   python run_dartree.py \
     --target-model Qwen/Qwen3-4B \
     --draft-model Huang2020/Qwen3-4B-Domino-b16 \
     --dataset gsm8k \
     --variant pruned \
     --temperature 0 \
     --device npu:0
   ```

If `torch_npu` is not installed, the code still runs unchanged on NVIDIA by
selecting a CUDA device (`--device cuda:0`, the default).

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

## Optional n-gram continuity scoring

DARTree can add DART's n-gram continuity term to tree scoring (Algorithm 1,
`w_level * (w_logit * logit + w_ngram * log(Pr_ngram(t | ctx) + eps))`):

```bash
python run_dartree.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model Huang2020/Qwen3-4B-Domino-b16 \
  --dataset gsm8k \
  --variant pruned \
  --temperature 0 \
  --ngram-model /path/to/small.trie \
  --ngram-weight 0.5
```

* `--ngram-model` is a DART-format `.trie` file loaded through the C++
  `TrieNgram` extension (`utils/ngram_cpp`, binary-compatible with DART's
  format). Published models are available at
  [hf.co/fvliang/dart-qwen3-ngram](https://huggingface.co/fvliang/dart-qwen3-ngram)
  (`full.trie`, and `small.trie` for fast testing — start with `small.trie`).
* `--ngram-weight` sets the weight of the n-gram term; `0.5` is DART's
  default, `0` disables it (ablation baseline). When `> 0`, `--ngram-model`
  is required.
* `--update-ngram` (optional) enables online adaptation: after each round's
  verification, the target-accepted token segment (lookback token + accepted
  chain + next token) is fed back into the loaded `.trie` in memory, so
  later rounds score against updated counts. The trie is only mutated in
  memory and is never saved to disk, so the on-disk model is untouched.
* The C++ extension is JIT-compiled on first use via
  `torch.utils.cpp_extension.load` (requires a C++20 compiler with OpenMP on
  the host) and is then cached under `~/.cache/torch_extensions`.
* To build your own `.trie` from JSONL (`{"text": ...}` per line) or from a
  DARTree dataset, use:

  ```bash
  python utils/ngram_build.py --data /path/to/jsonl_dir \
    --output-path /path/to/out --ngram-order 3 --n-jobs 16
  ```
