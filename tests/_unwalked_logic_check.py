"""Standalone logic test for eval_dartree._collect_unwalked, unwalked_summary
and save_unwalked (no torch needed).

Covers: fixed-variant tree (walked_past siblings + unreached descendants),
root-only acceptance, pruned-variant new<->old parent mapping, skip paths
(missing level record / token not in the parent's candidate row), summary
counts, and the CSV writer.
"""
import ast
import csv
import math
import tempfile
from pathlib import Path
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
names = ["_collect_unwalked", "unwalked_summary", "save_unwalked"]
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
collect = ns["_collect_unwalked"]
summary = ns["unwalked_summary"]
save = ns["save_unwalked"]


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
# Accepted path [0, 1]: node2 = walked_past sibling of the chain,
# node3 = walked_past (parent on path), node4 = unreached (parent node2
# was never walked).
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
    "fixed tree: walked_past + unreached",
    out,
    [
        {
            "out_pos": 101, "node": 2, "depth": 1,
            "token": 12, "parent_node": 0, "parent_token": 100,
            "category": "walked_past",
            "draft_rank": 2, "draft_prob": math.exp(-0.5),
            "ngram_rank": 2, "ngram_prob": 0.2, "ngram_order": 3,
        },
        {
            "out_pos": 102, "node": 3, "depth": 2,
            "token": 21, "parent_node": 1, "parent_token": 11,
            "category": "walked_past",
            "draft_rank": 1, "draft_prob": math.exp(-0.2),
            "ngram_rank": 2, "ngram_prob": 0.3, "ngram_order": 3,
        },
        {
            "out_pos": 102, "node": 4, "depth": 2,
            "token": 22, "parent_node": 2, "parent_token": 12,
            "category": "unreached",
            "draft_rank": 1, "draft_prob": math.exp(-0.3),
            # ngram row [0.1, 0.9, 0.2]: 0.9 and 0.2 are strictly better
            "ngram_rank": 3, "ngram_prob": 0.1, "ngram_order": 2,
        },
    ],
)

# Root-only acceptance: every final-tree node is a walked-past sibling.
out = run(
    accepted=[0],
    parents=[-1, 0, 0],
    depths=[1, 1],
    vid_tokens=[100, 11, 12],
    stats={"rank_pairs_detail": [LEVELS[1]]},
)
check(
    "root-only acceptance -> all walked_past",
    out,
    [
        {
            "out_pos": 101, "node": 1, "depth": 1,
            "token": 11, "parent_node": 0, "parent_token": 100,
            "category": "walked_past",
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_rank": 1, "ngram_prob": 0.5, "ngram_order": 3,
        },
        {
            "out_pos": 101, "node": 2, "depth": 1,
            "token": 12, "parent_node": 0, "parent_token": 100,
            "category": "walked_past",
            "draft_rank": 2, "draft_prob": math.exp(-0.5),
            "ngram_rank": 2, "ngram_prob": 0.2, "ngram_order": 3,
        },
    ],
)

# Pruned variant: new node 2/3 have old parents via kept_old (new 1 <-> old 3).
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
            "out_pos": 102, "node": 2, "depth": 2,
            "token": 51, "parent_node": 1, "parent_token": 31,
            "category": "walked_past",
            "draft_rank": 1, "draft_prob": math.exp(-0.2),
            "ngram_rank": 1, "ngram_prob": 0.5, "ngram_order": 3,
        },
        {
            "out_pos": 102, "node": 3, "depth": 2,
            "token": 52, "parent_node": 1, "parent_token": 31,
            "category": "walked_past",
            "draft_rank": 2, "draft_prob": math.exp(-0.8),
            "ngram_rank": 2, "ngram_prob": 0.4, "ngram_order": 3,
        },
    ],
)

# Skip paths: token not in the parent's candidate row; level record missing.
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
            "out_pos": 101, "node": 1, "depth": 1,
            "token": 11, "parent_node": 0, "parent_token": 100,
            "category": "walked_past",
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_rank": 1, "ngram_prob": 0.5, "ngram_order": 3,
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
    {"category": "walked_past"},
    {"category": "walked_past"},
    {"category": "unreached"},
])
check(
    "summary counts + frac_walked_past",
    s,
    {"total": 3.0, "walked_past": 2.0, "unreached": 1.0,
     "frac_walked_past": 2.0 / 3.0},
)
check("summary empty", summary([]),
      {"total": 0.0, "walked_past": 0.0, "unreached": 0.0,
       "frac_walked_past": 0.0})

# CSV writer.
csv_entries = [
    {"out_pos": 101, "node": 2, "depth": 1, "token": 12,
     "parent_node": 0, "parent_token": 100, "category": "walked_past",
     "draft_rank": 2, "draft_prob": math.exp(-0.5),
     "ngram_rank": 2, "ngram_prob": 0.2, "ngram_order": 3},
    {"out_pos": 102, "node": 4, "depth": 2, "token": 22,
     "parent_node": 2, "parent_token": 12, "category": "unreached",
     "draft_rank": 1, "draft_prob": math.exp(-0.3),
     "ngram_rank": 2, "ngram_prob": 0.1, "ngram_order": 2},
]
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "out.csv"
    save(csv_entries, p)
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    assert rows[0] == [
        "out_pos", "node", "depth", "token", "token_text",
        "parent_node", "parent_token", "parent_token_text", "category",
        "draft_rank", "draft_prob", "ngram_rank", "ngram_prob",
        "ngram_order",
    ]
    assert rows[1] == ["101", "2", "1", "12", "", "0", "100", "",
                       "walked_past", "2", "0.606531", "2", "0.2", "3"]
    assert rows[2] == ["102", "4", "2", "22", "", "2", "12", "",
                       "unreached", "1", "0.740818", "2", "0.1", "2"]
    assert len(rows) == 3
    # with a tokenizer stub
    class Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"
    save(
        [{"out_pos": 101, "node": 2, "depth": 1, "token": 12,
          "parent_node": 0, "parent_token": 100,
          "category": "walked_past", "draft_rank": 2,
          "draft_prob": 0.5, "ngram_rank": 2, "ngram_prob": 0.2,
          "ngram_order": 3}],
        p,
        Tok(),
    )
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    assert rows[1] == ["101", "2", "1", "12", "<12>", "0", "100", "<100>",
                       "walked_past", "2", "0.5", "2", "0.2", "3"]

print("\nALL UNWALKED CHECKS PASSED")
