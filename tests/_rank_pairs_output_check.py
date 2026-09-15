"""Standalone smoke check for eval_dartree.rank_pair_summary / save_rank_pairs
(no numpy / matplotlib / torch needed; exec'd from source with stubs)."""
import ast
import csv
import tempfile
from pathlib import Path
from typing import Any

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
funcs = {
    n.name: n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef)
    and n.name in ("rank_pair_summary", "save_rank_pairs")
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
save = ns["save_rank_pairs"]

pairs = [
    {"out_pos": 10, "token": 123, "draft_rank": 1, "ngram_rank": 3},
    {"out_pos": 11, "token": 456, "draft_rank": 2, "ngram_rank": 2},
    {"out_pos": 12, "token": 789, "draft_rank": 3, "ngram_rank": 1},
    {"out_pos": 13, "token": 111, "draft_rank": 4, "ngram_rank": 4},
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
    [{"out_pos": i, "token": i, "draft_rank": 2, "ngram_rank": 2} for i in range(5)]
)
assert s_const["n"] == 5.0 and s_const["pearson_rank_corr"] == 0.0

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / "out.json"
    csv_path = base.with_suffix(".rank_pairs.csv")
    png_path = base.with_suffix(".rank_pairs.png")
    save(pairs, csv_path, png_path, tokenizer=None)
    assert csv_path.exists(), "CSV must be written even without matplotlib"
    assert not png_path.exists(), "PNG must be skipped when matplotlib is absent"
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["out_pos", "token", "token_text", "draft_rank", "ngram_rank"]
    assert rows[1] == ["10", "123", "", "1", "3"]
    assert len(rows) == 5

print("\nALL RANK-PAIRS OUTPUT CHECKS PASSED")
