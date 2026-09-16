"""Standalone test for eval_dartree._renormalize_ngram_rows (no torch / numpy
needed) plus a static check that run_dartree.py defines and forwards
--renorm-ngram."""
import ast
import math


def close(a, b, tol=1e-12):
    return all(
        math.isclose(x, y, rel_tol=tol, abs_tol=tol)
        for x, y in zip(a, b)
    )

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef)
    and n.name == "_renormalize_ngram_rows"
)


class RowTensor:
    """Minimal tensor stand-in: sum(dim=-1, keepdim) / clamp_min / truediv
    with broadcasting of a [rows x 1] denominator."""

    def __init__(self, rows):
        self.rows = rows

    def sum(self, dim=-1, keepdim=False):
        assert dim == -1 and keepdim
        return RowTensor([[sum(r)] for r in self.rows])

    def clamp_min(self, eps):
        return RowTensor([[max(v, eps) for v in r] for r in self.rows])

    def __truediv__(self, other):
        out = []
        for r, o in zip(self.rows, other.rows):
            out.append([a / o[0] for a in r])
        return RowTensor(out)

    def __eq__(self, other):
        return len(self.rows) == len(other.rows) and all(
            close(r, o) for r, o in zip(self.rows, other.rows)
        )


ns = {"torch": type("TorchStub", (), {"Tensor": object})()}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<fn>", "exec"), ns)
renorm = ns["_renormalize_ngram_rows"]

# mixed row normalizes to sum 1; all-zero row stays zero (no NaN)
got = renorm(RowTensor([[0.3, 0.1, 0.0], [0.0, 0.0, 0.0]]), eps=1e-10)
want = RowTensor([[0.75, 0.25, 0.0], [0.0, 0.0, 0.0]])
assert got == want, f"got={got.rows} want={want.rows}"
print("OK  mixed row renormalized, all-zero row stays zero")

# single nonzero element -> 1.0
got = renorm(RowTensor([[0.0, 0.8]]), eps=1e-10)
assert got == RowTensor([[0.0, 1.0]]), got.rows
print("OK  single nonzero element -> 1.0")

# all-zero row with a healthy row: denominator clamp keeps it at zero
got = renorm(RowTensor([[0.0, 0.0], [0.5, 0.5]]), eps=1e-10)
assert got == RowTensor([[0.0, 0.0], [0.5, 0.5]]), got.rows
print("OK  all-zero row guarded against division by zero")

# static check: run_dartree.py must define and forward --renorm-ngram
rsrc = open("run_dartree.py", encoding="utf-8").read()
rtree = ast.parse(rsrc)
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "add_argument"
    and any(
        isinstance(a, ast.Constant) and a.value == "--renorm-ngram"
        for a in n.args
    )
    for n in ast.walk(rtree)
), "run_dartree.py must define --renorm-ngram"
assert 'engine_args.append("--renorm-ngram")' in rsrc, (
    "run_dartree.py must forward --renorm-ngram"
)
print("OK  run_dartree.py defines + forwards --renorm-ngram")

print("\nALL RENORM-NGRAM CHECKS PASSED")
