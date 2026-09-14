"""Standalone logic test for eval_dartree._ngram_contexts (no torch needed)."""
import ast
import sys

SRC = open("eval_dartree.py", encoding="utf-8").read()
tree = ast.parse(SRC)
fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "_ngram_contexts"
)
ns = {"torch": type("TorchStub", (), {"Tensor": object})()}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<fn>", "exec"), ns)
f = ns["_ngram_contexts"]


class T:
    """Minimal tensor stub: plain list with [i], tolist()."""

    def __init__(self, data):
        self.data = list(data)

    def __getitem__(self, i):
        return self.data[i]

    def tolist(self):
        return list(self.data)


def check(name, got, want):
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        raise SystemExit(f"FAILED: {name}")


# node ids: 0=root, 1=depth1, 2=depth2, 3=depth3
# parents_t[i] = parent of node i
tokens = [11, 12, 13, 14]  # tokens_t[i] = token id of node i+1 (no pad)
parents = [-1, 0, 1, 2, 3]

ROOT = 100
PREV = 200


def ctx(child_depth, parent_ids, order, prev=PREV):
    return f(
        child_depth=child_depth,
        parent_indices=T(parent_ids),
        tokens_t=T(tokens),
        parents_t=T(parents),
        root_token_id=ROOT,
        prev_root_token_id=prev,
        order=order,
    )


# order=3 (eval's trie): ctx_len = 2
check("d1/order3", ctx(1, [0], 3), [[PREV, ROOT]])
check("d2/order3", ctx(2, [1], 3), [[ROOT, 11]])       # grandparent = root
check("d3/order3", ctx(3, [2], 3), [[11, 12]])         # grandparent = node1
check("multi/order3", ctx(3, [2, 3], 3), [[11, 12], [12, 13]])

# order=2 (a 2-gram trie): ctx_len = 1
check("d1/order2", ctx(1, [0], 2), [[ROOT]])
check("d3/order2", ctx(3, [2], 2), [[12]])

# order=4: ctx_len = 3
check("d1/order4", ctx(1, [0], 4), [[PREV, ROOT]])
check("d2/order4", ctx(2, [1], 4), [[PREV, ROOT, 11]])  # pad root then prev
check("d3/order4", ctx(3, [2], 4), [[ROOT, 11, 12]])

# prev_root_token_id=None (default, e.g. fixed variant): no prev padding
check("no-prev/d1", ctx(1, [0], 3, prev=None), [[ROOT]])
check("no-prev/d3", ctx(3, [2], 3, prev=None), [[11, 12]])

print("\nALL NGRAM_CONTEXTS LOGIC TESTS PASSED")
