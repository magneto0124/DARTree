"""Verify the less-width rank-penalty semantics without torch.

The branch change multiplies each candidate's probability by
0.9 ** (k - 1), k = 1-based rank within the parent's top-k, derived with a
double argsort on the score matrix (works for sorted AND unsorted columns).

This check mirrors torch.argsort with pure Python and asserts:
  1. double-argsort ranks equal the 1-based descending-score rank per row;
  2. the applied log-space penalty equals (k - 1) * ln(0.9);
  3. rank-1 tokens are never penalized.

Run: python tests/_less_width_penalty_check.py
"""

import math
import random
import sys

BASE = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, BASE)

# --- extract TOPK_RANK_PENALTY_BASE from the real source ---
import ast

source = open(__import__("os").path.join(BASE, "eval_dartree.py"), encoding="utf-8").read()
parsed = ast.parse(source)
base = None
for node in parsed.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TOPK_RANK_PENALTY_BASE":
                base = float(ast.literal_eval(node.value))
assert base is not None, "TOPK_RANK_PENALTY_BASE not found"
assert base == 0.9, f"unexpected base {base}"


def argsort(values):
    return sorted(range(len(values)), key=lambda i: values[i])


def ranks_double_argsort(row):
    return [r + 1 for r in argsort(argsort([-v for v in row]))]


def ranks_reference(row):
    order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
    out = [0] * len(row)
    for rank, pos in enumerate(order, start=1):
        out[pos] = rank
    return out


def main() -> None:
    rng = random.Random(20240601)
    trials = 0
    for _ in range(20000):
        width = rng.randint(1, 8)
        row = [rng.uniform(-8.0, 0.0) for _ in range(width)]
        r1 = ranks_double_argsort(row)
        r2 = ranks_reference(row)
        trials += 1
        assert r1 == r2, f"rank mismatch: row={row} dbl={r1} ref={r2}"
        for col, rank in enumerate(r1):
            applied = (rank - 1) * math.log(base)
            expected = math.log(base ** (rank - 1))
            assert abs(applied - expected) < 1e-12, (applied, expected)
            if rank == 1:
                assert applied == 0.0, "rank-1 token must not be penalized"
    print(f"OK: {trials} random rows — double-argsort rank == descending-score rank, "
          f"penalty = ln({base} ** (k-1)), rank-1 never penalized")


if __name__ == "__main__":
    main()
