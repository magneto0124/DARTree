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

## Graft variant (retrieval-grafted hybrid trees)

`--variant graft` implements the *prune-then-graft* hybrid tree construction of
**"Draft Less, Retrieve More: Hybrid Tree Construction for Speculative
Decoding"** (Shen et al., 2026), adapted to DARTree's DFlash-style block
drafter.  The draft tree is expanded as usual (a wide supertree with
`--supertree-width` nodes per layer), but at calibrated pruning checkpoints
(`--prune-checkpoints`, depth `0` = the root itself) the *confidence* of the
highest-scoring draft path is compared against a threshold
(`--prune-thresholds`).  When it drops below the threshold, draft expansion
stops, only the top `fraction * budget` draft nodes are kept
(`--stage-draft-fractions`), and the released verification slots are grafted
with tokens retrieved from a GPU-resident adjacency matrix.

```bash
python run_dartree.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model Huang2020/Qwen3-4B-Domino-b16 \
  --dataset gsm8k \
  --variant graft \
  --temperature 0
```

Retrieval machinery (`utils/retrieval.py`):

* `RetrievalAdjacencyMatrix` — a GPU-resident `V x k` top-k successor table
  (paper Eq. 9).  Rows are updated online from target-model verification
  logits over the whole verified tree, accepted and rejected nodes alike
  (paper Eq. 12/14), plus the target prefill logits and the block drafter's
  target-head logits at the root (warm start; disable with
  `--no-graft-init-from-draft-logits`).
* `RetrievalTemplate` — a static, unbalanced template over successor ranks
  (paper Appendix A.1): the greedy top-1 continuation chain runs deep while
  lower-ranked alternatives get fewer descendants.  Each depth is
  materialized with one batched GPU gather, so the retrieval critical path
  scales with template depth, not node count.  Nodes whose parent row is
  still empty in the matrix are dropped prefix-closed, so a cold-start matrix
  degrades gracefully to the plain DARTree.
* `GraftConfig` — checkpoints, thresholds and stage draft/retrieval budget
  fractions.  Defaults mirror the paper's 60-node example (stages keep 8/24/40
  draft nodes and assign 52/36/20 slots to retrieval); for a `--tree-budget`
  of 64 the fractions become 8/26/43 draft nodes.

The final hybrid tree keeps the original verification budget (`|T_A| =
B_max`), is flattened into the standard tree-attention verification path, and
stays lossless: retrieved tokens are proposals that must still be accepted by
the target model under the standard speculative-decoding rule.  If no
checkpoint triggers, the variant degenerates to the plain pruned DARTree.
Per-round graft statistics (stage, retained draft nodes, retrieved nodes) are
reported via `--record-round-trace` and in the tree-stat totals.

**Calibration note:** the paper calibrates `--prune-thresholds` on a
warm-up set (ECHO-style).  The shipped defaults are a reasonable starting
point for greedy decoding; tune them on your workload.  A corpus warm-up for
the adjacency matrix (e.g. from ShareGPT, as in the paper) is not bundled —
rows are populated from prefill, root draft logits and online verification
updates instead.
