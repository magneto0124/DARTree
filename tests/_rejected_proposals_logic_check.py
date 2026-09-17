"""Standalone logic test for eval_dartree._collect_rejected_proposals,
rejected_proposal_summary and save_rejected_proposals (no torch needed).

Covers: root-only rejection, not_proposed (token never in the parent's
top-k row), not_expanded (parent at max depth), level_pruned (proposed but
not selected at this level), final_pruned (selected into the pre-prune tree
but dropped by the whole-tree prune), pruned-variant new<->old parent
mapping, missing ngram tables, and the CSV writer.
"""
import ast
import csv
import math
import tempfile
from pathlib import Path
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
names = ["_collect_rejected_proposals", "rejected_proposal_summary",
         "save_rejected_proposals"]
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
collect = ns["_collect_rejected_proposals"]
summary = ns["rejected_proposal_summary"]
save = ns["save_rejected_proposals"]


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


def run(next_token, accepted, depths, vid_tokens, stats, start=100):
    sink: list[dict[str, Any]] = []
    collect(
        next_token=next_token,
        accepted_indices=accepted,
        node_depths=depths,
        verify_input_ids=VID(vid_tokens),
        tree_stats=stats,
        start=start,
        sink=sink,
    )
    return sink


def L1(**extra):
    rec = {
        "child_depth": 1,
        "parent_ids": [0],
        "cands": [[11, 12, 13]],
        "draft_logprobs": [[-0.1, -0.5, -1.0]],
        "ngram_probs": [[0.5, 0.2, 0.0]],
        "ngram_matched": [[2, 2, 0]],
        "base_node_id": 0,
        "selected_pairs": [[0, 0]],
    }
    rec.update(extra)
    return rec


# 1) Root-only rejection: the parent is the root; the token WAS in the
#    root's top-k row but not selected at level 1 -> level_pruned.
out = run(
    next_token=12,
    accepted=[0],
    depths=[],
    vid_tokens=[100],
    stats={"rank_pairs_detail": [L1()]},
)
check(
    "root-only, proposed-but-not-selected -> level_pruned",
    out,
    [
        {
            "output_pos": 101, "depth": 1,
            "parent_node": 0, "parent_token": 100,
            "outcome": "level_pruned", "token": 12,
            "draft_rank": 2, "draft_prob": math.exp(-0.5),
            "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.2,
        }
    ],
)

# 2) Token never in the parent's top-k row -> not_proposed.
out = run(
    next_token=99,
    accepted=[0],
    depths=[],
    vid_tokens=[100],
    stats={"rank_pairs_detail": [L1()]},
)
check(
    "token absent from parent's top-k -> not_proposed",
    out,
    [
        {
            "output_pos": 101, "depth": 1,
            "parent_node": 0, "parent_token": 100,
            "outcome": "not_proposed", "token": 99,
        }
    ],
)

# 3) Parent at max depth: no level record for depth+1 -> not_expanded.
out = run(
    next_token=12,
    accepted=[0, 1],
    depths=[1, 1],
    vid_tokens=[100, 11],
    stats={"rank_pairs_detail": [L1()]},
)
check(
    "parent at max depth -> not_expanded",
    out,
    [
        {
            "output_pos": 102, "depth": 2,
            "parent_node": 1, "parent_token": 11,
            "outcome": "not_expanded", "token": 12,
        }
    ],
)

# 4) Root-only with no level-1 record -> not_expanded.
out = run(
    next_token=12,
    accepted=[0],
    depths=[],
    vid_tokens=[100],
    stats={"rank_pairs_detail": []},
)
check(
    "no level records at all -> not_expanded",
    out,
    [
        {
            "output_pos": 101, "depth": 1,
            "parent_node": 0, "parent_token": 100,
            "outcome": "not_expanded", "token": 12,
        }
    ],
)

# 5) Pruned variant: new id 1 <-> old id kept_old[0]=3; the token was
#    selected at level 2 (became a pre-prune node) -> final_pruned.
L2 = {
    "child_depth": 2,
    "parent_ids": [3, 2],
    "cands": [[51, 52, 53], [61, 62, 63]],
    "draft_logprobs": [[-0.2, -0.8, -1.0], [-0.3, -0.9, -1.1]],
    "ngram_probs": [[0.5, 0.4, 0.1], [0.1, 0.8, 0.2]],
    "ngram_matched": [[2, 2, 1], [1, 2, 1]],
    "base_node_id": 4,
    "selected_pairs": [[0, 0], [1, 1]],
}
out = run(
    next_token=51,
    accepted=[0, 1],
    depths=[1, 1],
    vid_tokens=[100, 31, 41],
    stats={"rank_pairs_detail": [L2], "prune_kept_old": [3, 7]},
)
check(
    "selected into pre-prune tree -> final_pruned",
    out,
    [
        {
            "output_pos": 102, "depth": 2,
            "parent_node": 1, "parent_token": 31,
            "outcome": "final_pruned", "token": 51,
            "draft_rank": 1, "draft_prob": math.exp(-0.2),
            "ngram_order": 3, "ngram_rank": 1, "ngram_prob": 0.5,
        }
    ],
)

# 6) Same level, token proposed but not selected -> level_pruned, with a
#    bigram match (matched len 1 -> ngram_order 2).
out = run(
    next_token=53,
    accepted=[0, 1],
    depths=[1, 1],
    vid_tokens=[100, 31, 41],
    stats={"rank_pairs_detail": [L2], "prune_kept_old": [3, 7]},
)
check(
    "proposed, bigram match, not selected -> level_pruned",
    out,
    [
        {
            "output_pos": 102, "depth": 2,
            "parent_node": 1, "parent_token": 31,
            "outcome": "level_pruned", "token": 53,
            "draft_rank": 3, "draft_prob": math.exp(-1.0),
            "ngram_order": 2, "ngram_rank": 3, "ngram_prob": 0.1,
        }
    ],
)

# 7) ngram table absent -> ngram fields default (0), draft fields kept.
out = run(
    next_token=11,
    accepted=[0],
    depths=[],
    vid_tokens=[100],
    stats={
        "rank_pairs_detail": [
            {
                "child_depth": 1,
                "parent_ids": [0],
                "cands": [[11, 12]],
                "draft_logprobs": [[-0.1, -0.2]],
                "base_node_id": 0,
                "selected_pairs": [[0, 0]],
            }
        ]
    },
)
check(
    "no ngram data -> ngram defaults, draft stats kept",
    out,
    [
        {
            "output_pos": 101, "depth": 1,
            "parent_node": 0, "parent_token": 100,
            "outcome": "final_pruned", "token": 11,
            "draft_rank": 1, "draft_prob": math.exp(-0.1),
            "ngram_order": 0, "ngram_rank": 0, "ngram_prob": 0.0,
        }
    ],
)

# 8) Summary counts.
entries = [
    {"outcome": "level_pruned"},
    {"outcome": "final_pruned"},
    {"outcome": "not_proposed"},
    {"outcome": "not_proposed"},
    {"outcome": "not_expanded"},
]
s = summary(entries)
check(
    "summary counts + frac_proposed",
    s,
    {
        "total": 5.0,
        "not_proposed": 2.0,
        "level_pruned": 1.0,
        "final_pruned": 1.0,
        "not_expanded": 1.0,
        "frac_proposed": 0.4,
    },
)
check("summary empty", summary([]),
      {"total": 0.0, "not_proposed": 0.0, "level_pruned": 0.0,
       "final_pruned": 0.0, "not_expanded": 0.0, "frac_proposed": 0.0})

# 9) CSV writer: exact column order per spec; token / parent_token are the
#    DECODED token text, not the token ids; parent node id sits before its
#    token.
csv_entries = [
    {"output_pos": 101, "depth": 1,
     "parent_node": 0, "parent_token": 100,
     "outcome": "level_pruned", "token": 12,
     "draft_rank": 2, "draft_prob": math.exp(-0.5),
     "ngram_order": 3, "ngram_rank": 2, "ngram_prob": 0.2},
    {"output_pos": 102, "depth": 2,
     "parent_node": 1, "parent_token": 31,
     "outcome": "not_proposed", "token": 99},
]


class Tok:
    def decode(self, ids):
        return f"<tok:{ids[0]}>"


with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "out.csv"
    save(csv_entries, p, Tok())
    rows = list(csv.reader(p.open(encoding="utf-8", newline="")))
    assert rows[0] == [
        "output_pos", "depth", "parent_node", "parent_token", "outcome",
        "token",
        "draft_rank", "draft_prob", "ngram_order", "ngram_rank",
        "ngram_prob",
    ]
    assert rows[1] == ["101", "1", "0", "<tok:100>", "level_pruned",
                       "<tok:12>",
                       "2", "0.606531", "3", "2", "0.2"]
    assert rows[2] == ["102", "2", "1", "<tok:31>", "not_proposed",
                       "<tok:99>",
                       "0", "0", "0", "0", "0"]
    assert len(rows) == 3

print("\nALL REJECTED-PROPOSALS CHECKS PASSED")
