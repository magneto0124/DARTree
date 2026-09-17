"""Verify the NNT correction semantics without torch.

The branch change re-scores a parent's top-k child candidates under the
grandparent's (next-next-token) distribution and blends the two conditional
probabilities in probability space:

    s'(c) = log(λ·p(c|p) + (1-λ)·p(c|g)),   λ = NNT_MIX_LAMBDA

Computed stably as m + log(λ·e^{a-m} + (1-λ)·e^{b-m}) with m = max(a, b),
where the grandparent log-prob for each candidate token is aligned to its
position in the shared (descending) candidate table via searchsorted on the
negated ids.

This check mirrors the tensor code with pure Python and asserts:
  1. the stable blend equals the direct formula log(λ·e^a + (1-λ)·e^b);
  2. the table-position alignment (negated binary search) selects the SAME
     token's grandparent probability as a direct per-token lookup;
  3. λ=1 recovers the parent scores and λ=0 recovers the grandparent scores;
  4. NNT_MIX_LAMBDA is read from the real source.

Run: python tests/_nnt_correction_check.py
"""

import ast
import math
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

source = open(os.path.join(BASE, "eval_dartree.py"), encoding="utf-8").read()
parsed = ast.parse(source)
LAMBDA = None
for node in parsed.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "NNT_MIX_LAMBDA":
                LAMBDA = float(ast.literal_eval(node.value))
assert LAMBDA is not None, "NNT_MIX_LAMBDA not found in eval_dartree.py"


def stable_blend(a, b, lam):
    m = max(a, b)
    return m + math.log(lam * math.exp(a - m) + (1.0 - lam) * math.exp(b - m))


def neg_searchsorted(sorted_desc, value):
    """Mirror torch.searchsorted(-table, -value) on a descending table."""
    neg_table = [-v for v in sorted_desc]
    neg_value = -value
    lo, hi = 0, len(neg_table)
    while lo < hi:
        mid = (lo + hi) // 2
        if neg_table[mid] < neg_value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main() -> None:
    rng = random.Random(20240617)
    trials = 0
    for _ in range(30000):
        table_size = rng.randint(2, 12)
        # descending candidate-id table (like torch.topk output)
        table = sorted(rng.sample(range(1000, 100000), table_size), reverse=True)
        k = rng.randint(1, table_size)
        n_parents = rng.randint(1, 4)

        # parent scores (log-probs) for its own top-k tokens, token ids = table ids
        parent_rows = []
        for _p in range(n_parents):
            tokens = rng.sample(table, k)  # top-k tokens of this parent
            scores = [math.log(rng.uniform(1e-4, 1.0)) for _ in tokens]
            parent_rows.append((tokens, scores))

        # grandparent rows: log-prob of EVERY table token (table order)
        gp_rows = []
        for _g in range(2):
            raw = [rng.uniform(0.0, 1.0) for _ in table]
            z = sum(raw)
            gp_rows.append([math.log(v / z) for v in raw])

        # grandparent id per parent (each parent has one grandparent)
        gp_of = [rng.randrange(2) for _ in range(n_parents)]
        # unique-grandparent map like torch.unique(return_inverse=True)
        uniq = sorted(set(gp_of))
        inverse = [uniq.index(g) for g in gp_of]

        for lam in (LAMBDA, 1.0, 0.0):
            for p in range(n_parents):
                tokens, scores = parent_rows[p]
                g_row = gp_rows[gp_of[p]]
                # reference: direct per-token lookup of p(token | g)
                ref = {tok: gp_row[table.index(tok)] for tok, gp_row in
                       zip(table, [g_row] * len(table))}
                for tok, a in zip(tokens, scores):
                    b = ref[tok]
                    # 1) stable blend == direct formula
                    direct = math.log(lam * math.exp(a) + (1.0 - lam) * math.exp(b))
                    got = stable_blend(a, b, lam)
                    assert abs(got - direct) < 1e-12, (got, direct)
                    # 3) boundary lambdas
                    if lam == 1.0:
                        assert abs(got - a) < 1e-12
                    if lam == 0.0:
                        assert abs(got - b) < 1e-12
                    trials += 1

        # 2) searchsorted alignment: table_pos of each parent token matches
        #    table.index(), and gathering the grandparent log-prob at that
        #    position equals the direct lookup used above.
        for p in range(n_parents):
            tokens, scores = parent_rows[p]
            g_row = gp_rows[gp_of[p]]
            g_row_uniq_indexed = [g_row]  # per unique grandparent, table order
            for tok in tokens:
                pos = neg_searchsorted(table, tok)
                assert table[pos] == tok, (table, tok, pos)
                gathered = g_row_uniq_indexed[0][pos]
                direct = g_row[table.index(tok)]
                assert abs(gathered - direct) < 1e-15, (gathered, direct)
                trials += 1

    print(f"OK: {trials} checks — stable blend == direct mixture formula, "
          f"searchsorted alignment picks the same token's grandparent log-prob, "
          f"λ=1/λ=0 boundaries correct, NNT_MIX_LAMBDA={LAMBDA}")


if __name__ == "__main__":
    main()
