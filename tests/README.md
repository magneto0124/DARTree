# DARTree Graft tests

Tests for the retrieval (Graft) machinery added to DARTree:

| File | Covers |
|---|---|
| `test_retrieval_matrix.py` | `RetrievalAdjacencyMatrix` — the GPU-resident top-k successor table (paper Eq. 9): construction, online updates from target logits (Eq. 12/14), mean-pooling over duplicate parents, overwriting, lookups |
| `test_retrieval_template.py` | Retrieval-tree construction: `make_decaying_depth_counts` (imbalanced per-depth template sizes), `build_rank_template` (greedy chain + round-robin ranks), `RetrievalTemplate.materialize` (per-depth batched matrix gathers, prefix-closed cold-start dropping) |
| `test_graft_config.py` | `GraftConfig` — checkpoint/threshold/fraction validation and the stage-adaptive draft/retrieval budget split (Eq. 13) |
| `test_graft_merge.py` | The prune-then-graft fusion step: `select_topb_prefix_tree` (prefix-closed Top-B) plus a re-implementation of the merge block of `eval_dartree.build_dartree_supertree` asserting budget conservation, topological order, prefix closure, retrieval nodes hanging off the root, and token/rank consistency with the matrix |

## Requirements

* Python 3.10+
* `torch` (CPU-only is enough — all tests use `device="cpu"`)
* `transformers` (only `test_graft_merge.py`, which imports `eval_dartree`)

No models, GPU or network access are needed.

## Run

From the repository root:

```bash
python -m pytest tests/ -v
```

or a single file:

```bash
python -m pytest tests/test_retrieval_template.py -v
```

`tests/conftest.py` adds the repository root to `sys.path` so both
`import utils.retrieval` and `import eval_dartree` work.

## Notes

* `utils/retrieval.py` also ships a lightweight `_self_test()` runnable via
  `python -m utils.retrieval`; the pytest suite covers the same ground in a
  structured, assert-per-case form.
* The fusion tests deliberately re-implement the merge block of
  `eval_dartree.build_dartree_supertree` over the same building blocks rather
  than calling the builder directly (which would need full draft/target
  models).  If that block is ever refactored, keep the invariants asserted
  here in mind.
