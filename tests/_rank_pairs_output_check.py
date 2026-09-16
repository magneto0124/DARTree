"""Standalone smoke check for eval_dartree.rank_pair_summary and
_save_rank_scatter (no numpy / matplotlib / torch needed; exec'd from
source with stubs)."""
import ast
import tempfile
from pathlib import Path
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
funcs = {
    n.name: n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef)
    and n.name in ("rank_pair_summary", "_save_rank_scatter")
}


class _M:
    def __init__(self, v):
        self.v = v

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return self.v[k[0]][k[1]]
        return self.v[k]


class _A:
    def __init__(self, v):
        self.v = list(v)

    def __iter__(self):
        return iter(self.v)

    def __len__(self):
        return len(self.v)

    def mean(self):
        return sum(self.v) / len(self.v)

    def __lt__(self, other):
        return _A(a < b for a, b in zip(self.v, other.v))

    def __eq__(self, other):
        return _A(a == b for a, b in zip(self.v, other.v))


class NpStub:
    float64 = float

    @staticmethod
    def array(v, dtype=None):
        return _A(float(x) for x in v)

    @staticmethod
    def corrcoef(xs, ys):
        n = len(xs)
        if n <= 1:
            raise ValueError("corrcoef needs >1")
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
        vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
        vy = sum((y - my) ** 2 for y in ys) / (n - 1)
        c = cov / ((vx * vy) ** 0.5)
        return _M([[1.0, c], [c, 1.0]])

    @staticmethod
    def mean(v):
        return sum(v) / len(v)


ns = {"Any": Any, "Path": Path, "np": NpStub, "torch": type("T", (), {"Tensor": object})()}
for name, fn in funcs.items():
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"<{name}>", "exec"), ns)

summary = ns["rank_pair_summary"]
scatter = ns["_save_rank_scatter"]

# hit entries carry the tree-table keys; rank_pair_summary only reads ranks
pairs = [
    {"output_pos": 101, "depth": 1, "parent_token": 100,
     "category": "hit", "token": 11, "draft_rank": 1, "ngram_rank": 3},
    {"output_pos": 102, "depth": 2, "parent_token": 11,
     "category": "hit", "token": 21, "draft_rank": 2, "ngram_rank": 2},
    {"output_pos": 103, "depth": 3, "parent_token": 21,
     "category": "hit", "token": 31, "draft_rank": 3, "ngram_rank": 1},
    {"output_pos": 104, "depth": 4, "parent_token": 31,
     "category": "hit", "token": 41, "draft_rank": 4, "ngram_rank": 4},
]
s = summary(pairs)
print("summary:", s)
assert s["n"] == 4.0
assert s["mean_draft_rank"] == 2.5 and s["mean_ngram_rank"] == 2.5
# pairs: (1,3),(2,2),(3,1),(4,4) -> y<x only for (3,1): 0.25; y==x for 2 of 4
assert s["frac_ngram_better_than_draft"] == 0.25
assert s["frac_equal_rank"] == 0.5
assert abs(s["pearson_rank_corr"] - 0.2) < 1e-9

assert summary([]) == {"n": 0.0}

# constant series -> corrcoef would be NaN; guard must return 0.0
s_const = summary(
    [{"draft_rank": 2, "ngram_rank": 2} for _ in range(5)]
)
assert s_const["n"] == 5.0 and s_const["pearson_rank_corr"] == 0.0

# scatter: matplotlib absent -> skipped without crashing
with tempfile.TemporaryDirectory() as tmp:
    png = Path(tmp) / "scatter.png"
    scatter(pairs, png)
    assert not png.exists(), "PNG must be skipped when matplotlib is absent"
    scatter([], png)  # empty hit list must also be safe

print("\nALL RANK-SUMMARY/SCATTER CHECKS PASSED")
