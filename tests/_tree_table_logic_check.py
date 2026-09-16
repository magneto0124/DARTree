"""Standalone logic test for eval_dartree._collect_tree_table, tree_table_summary
and save_tree_table (no torch needed).

Covers: fixed-variant tree with hit / walked_past / unreached categories,
root-only acceptance (no hit), pruned-variant new<->old parent mapping, skip
paths (missing level record / token not in the parent's candidate row),
summary counts, the CSV writer (exact column order), and the matplotlib-less
scatter fallback.
"""
import ast
import csv
import math
import tempfile
from pathlib import Path
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
names = [
    "_collect_tree_table",
    "tree_table_summary",
    "save_tree_table",
    "_save_rank_scatter",
]
fns = [
    next(n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in names
]
ns = {
    "torch": type("TorchStub", (), {"Tensor": object})(),
    "Any": Any,
    "np": type("NpStub", (), {"exp": staticmethod(math.exp)})(),
    "csv": csv,
    "Path": Path,
}
exec(compile(ast.Module(body=fns, type_ignores=[]), "<fns>", "exec"), ns)
collect = ns["_collect_tree_table"]
summary = ns["tree_table_summary"]
save = ns["save_tree_table"]


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


# Tree: node1<-root, node2<-root, node3<-node1, node4<-node2.
# Accepted path [0, 1]: node1 = hit, node2 = walked_past, node3 = walked_past
# (parent on path), node4 = unreached (parent node2 never walked).
LEVELS = {
    1: {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12, 13]],
        "draft_logprobs": [[-0.1, -0.5, -1.0]],
        "ngram_probs": [[0.5, 0.2, 0.0]],
        "ngram_matched": [[2, 2, 0]],
    },
    2: {
        "child_depth": 2,
        "parent_ids": [1, 2],
        "cands": [[21, 22, 23], [22, 21, 24]],
        "draft_logprobs": [[-0.2, -0.6, -1.2], [-0.3, -0.4, -1.5]],
        "ngram_probs": [[0.3, 0.6, 0.0], [0.1, 0.9, 0.2]],
        "ngram_matched": [[2, 2, 0], [1, 2, 1]],
    },
}
out = run(
    accepted=[0, 1],
    parents=[-1, 0, 0, 1, 2],
    depths=[1, 1, 2, 2],
    vid_tokens=[100, 11, 12, 21, 22],
    stats={"rank_pairs_detail": list(LEVELS.values())},
)
check(
    "fixed tree: hit + walked_past + unreached",
    out,
    [
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "hit", "token": 11,
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.5,
        },
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "walked_past", "token": 12,
            "draft_rank": 2, "draft_prob": math.exp(-0.5),
            "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.2,
        },
        {
            "output_pos": 102, "depth": 2, "parent_token": 11,
            "category": "walked_past", "token": 21,
            "draft_rank": 1, "draft_prob": math.exp(-0.2),
            "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.3,
        },
        {
            "output_pos": 102, "depth": 2, "parent_token": 12,
            "category": "unreached", "token": 22,
            "draft_rank": 1, "draft_prob": math.exp(-0.3),
            # ngram row [0.1, 0.9, 0.2]: 0.9 and 0.2 are strictly better
            "ngram_order": 2, "ngram_rank": 3, "ngram_prob": 0.1,
        },
    ],
)

# Root-only acceptance: no hit, every node is a walked-past sibling.
out = run(
    accepted=[0],
    parents=[-1, 0, 0],
    depths=[1, 1],
    vid_tokens=[100, 11, 12],
    stats={"rank_pairs_detail": [LEVELS[1]]},
)
check(
    "root-only acceptance -> all walked_past, no hit",
    out,
    [
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "walked_past", "token": 11,
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.5,
        },
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "walked_past", "token": 12,
            "draft_rank": 2, "draft_prob": math.exp(-0.5),
            "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.2,
        },
    ],
)

# Pruned variant: new nodes 2/3 have old parent 3 via kept_old (new 1 <-> old 3).
out = run(
    accepted=[0, 1],
    parents=[-1, 0, 1, 1],
    depths=[1, 2, 2],
    vid_tokens=[100, 31, 51, 52],
    stats={
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
                "parent_ids": [3],
                "cands": [[51, 52, 53]],
                "draft_logprobs": [[-0.2, -0.8, -1.0]],
                "ngram_probs": [[0.5, 0.4, 0.1]],
                "ngram_matched": [[2, 2, 1]],
            },
        ],
        "prune_kept_old": [3, 7, 9],
    },
)
check(
    "pruned remap (old parent 3 via kept_old)",
    out,
    [
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "hit", "token": 31,
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.9,
        },
        {
            "output_pos": 102, "depth": 2, "parent_token": 31,
            "category": "walked_past", "token": 51,
            "draft_rank": 1, "draft_prob": math.exp(-0.2),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.5,
        },
        {
            "output_pos": 102, "depth": 2, "parent_token": 31,
            "category": "walked_past", "token": 52,
            "draft_rank": 2, "draft_prob": math.exp(-0.8),
            "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.4,
        },
    ],
)

# Skip paths: token not in the parent's candidate row; no level records.
out = run(
    accepted=[0],
    parents=[-1, 0, 0],
    depths=[1, 1],
    vid_tokens=[100, 11, 999],
    stats={"rank_pairs_detail": [LEVELS[1]]},
)
check(
    "token not in candidate row -> skipped",
    out,
    [
        {
            "output_pos": 101, "depth": 1, "parent_token": 100,
            "category": "walked_past", "token": 11,
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.5,
        }
    ],
)
out = run(
    accepted=[0],
    parents=[-1, 0, 0],
    depths=[1, 1],
    vid_tokens=[100, 11, 12],
    stats={"rank_pairs_detail": []},
)
check("no level records -> empty", out, [])

# Summary counts.
s = summary([
    {"category": "hit"},
    {"category": "hit"},
    {"category": "walked_past"},
    {"category": "unreached"},
])
check(
    "summary counts + frac_hit",
    s,
    {"total": 4.0, "hit": 2.0, "walked_past": 1.0, "unreached": 1.0,
     "frac_hit": 0.5},
)
check("summary empty", summary([]),
      {"total": 0.0, "hit": 0.0, "walked_past": 0.0, "unreached": 0.0,
       "frac_hit": 0.0})

# CSV writer: exact column order per spec; token / parent_token are the
# DECODED token text, not the token ids.
csv_entries = [
    {"output_pos": 101, "depth": 1, "parent_token": 100,
     "category": "hit", "token": 11,
     "draft_rank": 1, "draft_prob": 0.5,
     "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.9},
    {"output_pos": 101, "depth": 1, "parent_token": 100,
     "category": "walked_past", "token": 12,
     "draft_rank": 2, "draft_prob": math.exp(-0.5),
     "ngram_order": 2, "ngram_rank": 2, "ngram_prob": 0.2},
]


class Tok:
    def decode(self, ids):
        return f"<tok:{ids[0]}>"


with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "out.csv"
    save(csv_entries, p, Tok())
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    assert rows[0] == [
        "output_pos", "depth", "parent_token", "category", "token",
        "draft_rank", "draft_prob", "ngram_order", "ngram_rank",
        "ngram_prob",
    ]
    assert rows[1] == ["101", "1", "<tok:100>", "hit", "<tok:11>",
                       "1", "0.5", "3", "1", "0.9"]
    assert rows[2] == ["101", "1", "<tok:100>", "walked_past", "<tok:12>",
                       "2", "0.606531", "2", "2", "0.2"]
    assert len(rows) == 3
    # with a png path: matplotlib missing -> scatter skipped, no crash
    save(csv_entries, p, Tok(), png_path=Path(tmp) / "scatter.png")
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    assert len(rows) == 3

print("\nALL TREE-TABLE CHECKS PASSED")
