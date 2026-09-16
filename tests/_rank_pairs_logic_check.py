"""Standalone logic test for eval_dartree._collect_rank_pairs (no torch needed).

Covers: rank computation (strictly-better counting, ties share a rank),
per-candidate ngram match order (2-gram / 3-gram / no match) recorded via
``ngram_matched``, draft/ngram probability values, multi-parent / multi-level
alignment, pruned-variant old<->new node mapping, and the skip paths
(missing ngram table / unknown parent / unknown token).
"""
import ast
import math
import sys
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "_collect_rank_pairs"
)
ns = {
    "torch": type("TorchStub", (), {"Tensor": object})(),
    "Any": Any,
    "np": type("NpStub", (), {"exp": staticmethod(math.exp)})(),
}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<fn>", "exec"), ns)
collect = ns["_collect_rank_pairs"]


class VID:
    """Stub for verify_input_ids: vid[(0, node)] -> token id of node."""

    def __init__(self, tokens):
        self.tokens = list(tokens)  # index = node id

    def __getitem__(self, key):
        row, node = key
        assert row == 0
        return self.tokens[node]


def check(name, got, want):
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def run(accepted, parents, depths, vid_tokens, stats, start=100):
    sink: list[dict[str, Any]] = []
    collect(
        accepted_indices=accepted,
        parents=parents,
        node_depths=depths,
        verify_input_ids=VID(vid_tokens),
        tree_stats=stats,
        start=start,
        sink=sink,
    )
    return sink


# ---------------------------------------------------------------------------
# 1) Non-pruned: 2 levels, multi-parent, distinct draft/ngram orderings,
#    mixed ngram match orders (row0 uses a 2-gram match for the accepted
#    token 21; row1 mixes 3-gram/2-gram/no-match across its candidates).
#    node ids: 0=root, 1/2=depth1 (tokens 11/12), 3/4=depth2 (21/22).
# ---------------------------------------------------------------------------
LEVELS = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12, 13]],
        "draft_logprobs": [[-0.1, -0.5, -1.0]],
        "ngram_probs": [[0.5, 0.2, 0.0]],
        "ngram_matched": [[2, 2, 0]],  # 11,12: trigram; 13: no match
    },
    2: {
        "child_depth": 2,
        "parent_ids": [1, 2],
        "cands": [[21, 22, 23], [22, 21, 24]],
        "draft_logprobs": [[-0.2, -0.6, -1.2], [-0.3, -0.4, -1.5]],
        "ngram_probs": [[0.3, 0.6, 0.0], [0.1, 0.9, 0.2]],
        "ngram_matched": [[1, 2, 0], [2, 1, 1]],
    },
}
STATS = {"rank_pairs_detail": list(LEVELS.values())}
VID_TOKENS = [100, 11, 12, 21, 22]

out = run(
    accepted=[0, 1, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=VID_TOKENS,
    stats=STATS,
)
check(
    "non-pruned chain (incl. probs + ngram order)",
    out,
    [
        {
            "out_pos": 101, "token": 11,
            "depth": 1,
            "draft_rank": 1, "ngram_rank": 1,
            "draft_prob": math.exp(-0.1), "ngram_prob": 0.5,
            "ngram_order": 3,
        },
        {
            "out_pos": 102, "token": 21,
            "depth": 2,
            "draft_rank": 1, "ngram_rank": 2,
            "draft_prob": math.exp(-0.2), "ngram_prob": 0.3,
            "ngram_order": 2,  # bigram backoff for the accepted token
        },
    ],
)

# 2) Ties: strictly-better counting -> equal scores share the top rank;
#    no-match candidate gets ngram_order 0.
LEVELS_TIE = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[7, 8, 9]],
        "draft_logprobs": [[-0.5, -0.5, -1.0]],
        "ngram_probs": [[0.2, 0.2, 0.0]],
        "ngram_matched": [[1, 2, 0]],
    }
}
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 8],
    stats={"rank_pairs_detail": list(LEVELS_TIE.values())},
)
check(
    "tie -> shared rank 1, trigram order",
    out,
    [
        {
            "out_pos": 101, "token": 8,
            "depth": 1,
            "draft_rank": 1, "ngram_rank": 1,
            "draft_prob": math.exp(-0.5), "ngram_prob": 0.2,
            "ngram_order": 3,
        }
    ],
)
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 9],
    stats={"rank_pairs_detail": list(LEVELS_TIE.values())},
)
check(
    "tie -> third cand rank 3, no ngram match",
    out,
    [
        {
            "out_pos": 101, "token": 9,
            "depth": 1,
            "draft_rank": 3, "ngram_rank": 3,
            "draft_prob": math.exp(-1.0), "ngram_prob": 0.0,
            "ngram_order": 0,
        }
    ],
)

# 3) Pruned remap: new node i <-> old node kept_old[i-1] (root maps to 0).
KEPT = [5, 3, 7, 9, 11]
STATS_PRUNED = {
    "rank_pairs_detail": [
        {
            "child_depth": 1,
            "parent_ids": [0],
            "cands": [[31, 32, 33]],
            "draft_logprobs": [[-0.1, -0.2, -0.3]],
            "ngram_probs": [[0.9, 0.1, 0.0]],
            "ngram_matched": [[2, 1, 0]],
        },
        {
            "child_depth": 2,
            "parent_ids": [5, 3],
            "cands": [[51, 52, 53], [61, 62, 63]],
            "draft_logprobs": [[-0.2, -0.8, -1.0], [-0.3, -0.9, -1.1]],
            "ngram_probs": [[0.5, 0.4, 0.1], [0.1, 0.8, 0.2]],
            "ngram_matched": [[2, 2, 1], [1, 2, 1]],
        },
    ],
    "prune_kept_old": KEPT,
}
out = run(
    accepted=[0, 1, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=[100, 31, 41, 51, 61],
    stats=STATS_PRUNED,
)
check(
    "pruned remap chain (incl. probs + ngram order)",
    out,
    [
        {
            "out_pos": 101, "token": 31,
            "depth": 1,
            "draft_rank": 1, "ngram_rank": 1,
            "draft_prob": math.exp(-0.1), "ngram_prob": 0.9,
            "ngram_order": 3,
        },
        {
            "out_pos": 102, "token": 51,
            "depth": 2,
            "draft_rank": 1, "ngram_rank": 1,
            "draft_prob": math.exp(-0.2), "ngram_prob": 0.5,
            "ngram_order": 3,
        },
    ],
)

# 4) Skip paths.
#    a) level record without ngram data (ngram scoring disabled there)
no_ngram = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12]],
        "draft_logprobs": [[-0.1, -0.2]],
    }
}
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 11],
    stats={"rank_pairs_detail": list(no_ngram.values())},
)
check("missing ngram table -> skipped", out, [])

#    b) ngram table present but ngram_matched missing -> order 0, still recorded
no_matched = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12]],
        "draft_logprobs": [[-0.1, -0.2]],
        "ngram_probs": [[0.5, 0.1]],
    }
}
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 11],
    stats={"rank_pairs_detail": list(no_matched.values())},
)
check(
    "ngram_matched absent -> ngram_order 0",
    out,
    [
        {
            "out_pos": 101, "token": 11,
            "depth": 1,
            "draft_rank": 1, "ngram_rank": 1,
            "draft_prob": math.exp(-0.1), "ngram_prob": 0.5,
            "ngram_order": 0,
        }
    ],
)

#    c) accepted token absent from the parent's candidate table
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 999],
    stats=STATS,
)
check("token not in candidates -> skipped", out, [])

#    d) level record for the accepted node's depth is missing
out = run(
    accepted=[0, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=VID_TOKENS,
    stats={"rank_pairs_detail": [LEVELS[1]]},
)
check("depth record missing -> skipped", out, [])

#    e) root-only acceptance (nothing accepted): no chain nodes to record
out = run(
    accepted=[0],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=VID_TOKENS,
    stats=STATS,
)
check("root-only acceptance -> empty", out, [])

print("\nALL RANK-PAIRS LOGIC CHECKS PASSED")
