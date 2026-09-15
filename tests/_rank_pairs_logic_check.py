"""Standalone logic test for eval_dartree._collect_rank_pairs (no torch needed).

Covers: rank computation (strictly-better counting, ties share a rank),
multi-level / multi-parent alignment, pruned-variant old<->new node mapping,
and the skip paths (missing ngram table / unknown parent / unknown token).
"""
import ast
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
# 1) Non-pruned: 2 levels, multi-parent, distinct draft/ngram orderings.
#    node ids: 0=root, 1/2=depth1 (tokens 11/12), 3/4=depth2 (21/22).
# ---------------------------------------------------------------------------
LEVELS = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12, 13]],
        "draft_logprobs": [[-0.1, -0.5, -1.0]],
        "ngram_probs": [[0.5, 0.2, 0.0]],
    },
    2: {
        "child_depth": 2,
        "parent_ids": [1, 2],
        "cands": [[21, 22, 23], [22, 21, 24]],
        "draft_logprobs": [[-0.2, -0.6, -1.2], [-0.3, -0.4, -1.5]],
        "ngram_probs": [[0.3, 0.6, 0.0], [0.1, 0.9, 0.2]],
    },
}
STATS = {"rank_pairs_detail": list(LEVELS.values())}
# tokens: node 0..4
VID_TOKENS = [100, 11, 12, 21, 22]

# accepted: root -> node1(11, depth1) -> node3(21, depth2)
out = run(
    accepted=[0, 1, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],  # node_depths[node-1] = depth of node
    vid_tokens=VID_TOKENS,
    stats=STATS,
)
check(
    "non-pruned chain",
    out,
    [
        {"out_pos": 101, "token": 11, "draft_rank": 1, "ngram_rank": 1},
        {"out_pos": 102, "token": 21, "draft_rank": 1, "ngram_rank": 2},
    ],
)

# 2) Ties: strictly-better counting -> equal scores share the top rank.
LEVELS_TIE = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[7, 8, 9]],
        "draft_logprobs": [[-0.5, -0.5, -1.0]],
        "ngram_probs": [[0.2, 0.2, 0.0]],
    }
}
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 8],
    stats={"rank_pairs_detail": list(LEVELS_TIE.values())},
)
check("tie -> shared rank 1", out, [{"out_pos": 101, "token": 8, "draft_rank": 1, "ngram_rank": 1}])
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 9],
    stats={"rank_pairs_detail": list(LEVELS_TIE.values())},
)
check("tie -> third cand rank 3", out, [{"out_pos": 101, "token": 9, "draft_rank": 3, "ngram_rank": 3}])

# 3) Pruned remap: new node i <-> old node kept_old[i-1] (root maps to 0).
#    kept: new 1<->old5, new 2<->old3, new 3<->old7, new 4<->old9, new 5<->old11
KEPT = [5, 3, 7, 9, 11]
STATS_PRUNED = {
    "rank_pairs_detail": [
        {
            "child_depth": 1,
            "parent_ids": [0],  # depth-1 frontier is always the root (old 0)
            "cands": [[31, 32, 33]],
            "draft_logprobs": [[-0.1, -0.2, -0.3]],
            "ngram_probs": [[0.9, 0.1, 0.0]],
        },
        {
            "child_depth": 2,
            "parent_ids": [5, 3],  # old ids of new nodes 1, 2
            "cands": [[51, 52, 53], [61, 62, 63]],
            "draft_logprobs": [[-0.2, -0.8, -1.0], [-0.3, -0.9, -1.1]],
            "ngram_probs": [[0.5, 0.4, 0.1], [0.1, 0.8, 0.2]],
        },
    ],
    "prune_kept_old": KEPT,
}
# accepted: root -> new node 1 (old 5, token 31, depth1, parent new 0)
#                -> new node 3 (old 7, token 51, depth2, parent new 1 = old 5)
out = run(
    accepted=[0, 1, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=[100, 31, 41, 51, 61],
    stats=STATS_PRUNED,
)
check(
    "pruned remap chain",
    out,
    [
        {"out_pos": 101, "token": 31, "draft_rank": 1, "ngram_rank": 1},
        {"out_pos": 102, "token": 51, "draft_rank": 1, "ngram_rank": 1},
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
    depths=[0, 1],
    vid_tokens=[100, 11],
    stats={"rank_pairs_detail": list(no_ngram.values())},
)
check("missing ngram table -> skipped", out, [])

#    b) accepted token absent from the parent's candidate table
out = run(
    accepted=[0, 1],
    parents=[-1, 0],
    depths=[1],
    vid_tokens=[100, 999],
    stats=STATS,
)
check("token not in candidates -> skipped", out, [])

#    c) level record for the accepted node's depth is missing
out = run(
    accepted=[0, 3],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=VID_TOKENS,
    stats={"rank_pairs_detail": [LEVELS[1]]},  # only depth-1 recorded
)
check("depth record missing -> skipped", out, [])

#    d) root-only acceptance (nothing accepted): no chain nodes to record
out = run(
    accepted=[0],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=VID_TOKENS,
    stats=STATS,
)
check("root-only acceptance -> empty", out, [])

print("\nALL RANK-PAIRS LOGIC CHECKS PASSED")
