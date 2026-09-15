"""Extended standalone check for eval_dartree._ngram_contexts + the ngram
scoring block (lines ~645-665).

Simulates the exact buffer layouts that build_dartree_supertree produces
(level-by-level contiguous node numbering, tokens_t[j-1] = token of node j,
parents_t[j] = parent of node j, parents_t[0] = -1) across several levels with
arbitrary parent pointers into the previous frontier, then cross-checks the
function against an independent reference that computes the intended context
directly from (depth, path tokens, root, prev).  Also validates that the
candidate rows from top_ids are paired with the correct parent context.

No torch required (function is exec'd from source with a Tensor stub).
"""
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


def reference_contexts(child_depth, parent_indices, node_depth, node_token,
                       root, prev, order):
    """Independent reference: context = last (order-1) tokens BEFORE the
    candidate position (round-position child_depth), i.e. the parent's path
    tokens at round-positions child_depth-1 .. child_depth-order+1, mapped to
    the accepted prefix (root at position 0, prev at position -1) whenever the
    tree path runs out.  Oldest first (chronological)."""
    ctx_len = max(0, order - 1)
    out = []
    for p in parent_indices:
        # tokens available by round-position: tree path at positions 1..depth,
        # root at position 0, prev at position -1 (nothing older than -1).
        path = [root] + [node_token[n] for n in ancestors(p, node_depth)]
        available = (([prev] if prev is not None else []) + path)
        # context = the last up-to-ctx_len available tokens (chronological):
        # positions child_depth-1 .. child_depth-ctx_len, clipped at -1.
        start = max(0, len(available) - ctx_len)
        out.append(available[start:])
    return out


def ancestors(node, node_depth):
    """node ids on the path from the root's child down to `node`, oldest first."""
    # depth of node = node_depth[node]; walk up collecting ids
    chain = []
    n = node
    while n > 0:
        chain.append(n)
        n = parent_of[n]
    return list(reversed(chain))


# ---------------------------------------------------------------------------
# Build a realistic multi-level tree with the exact write pattern of
# build_dartree_supertree: level L nodes get ids (prev_count+1 .. prev_count+k)
# contiguously; tokens_t[j-1] = token of node j; parents_t[j] = parent id.
# ---------------------------------------------------------------------------
ROOT, PREV = 1000, 2000
ORDER = 3  # the eval trie is 3-gram -> ctx_len = 2

tokens_t = []          # grows; token of node j at index j-1
parents_t = [-1]       # parents_t[j] = parent of node j; parents_t[0] = -1
node_depth = {0: 0}
node_token = {0: ROOT}

# level 1: 3 nodes (ids 1,2,3), all children of root 0
level1 = [11, 12, 13]
for tok in level1:
    node_id = len(parents_t)
    tokens_t.append(tok)
    parents_t.append(0)
    node_depth[node_id] = 1
    node_token[node_id] = tok

# level 2: 4 nodes (ids 4..7), parents chosen arbitrarily from level 1
level2 = [(1, 21), (3, 22), (1, 23), (2, 24)]  # (parent_id, token)
for pid, tok in level2:
    node_id = len(parents_t)
    tokens_t.append(tok)
    parents_t.append(pid)
    node_depth[node_id] = 2
    node_token[node_id] = tok

# level 3: 3 nodes (ids 8..10), parents from level 2
level3 = [(5, 31), (4, 32), (7, 33)]
for pid, tok in level3:
    node_id = len(parents_t)
    tokens_t.append(tok)
    parents_t.append(pid)
    node_depth[node_id] = 3
    node_token[node_id] = tok

parent_of = {i: parents_t[i] for i in range(len(parents_t))}

print("== order=3 (ctx_len=2), multi-parent, arbitrary parent pointers ==")
for child_depth, parents in [
    (1, [0]),
    (2, [1, 2, 3]),
    (3, [4, 5, 6, 7]),
    (4, [8, 9, 10]),
]:
    got = f(child_depth=child_depth,
            parent_indices=T(parents),
            tokens_t=T(tokens_t), parents_t=T(parents_t),
            root_token_id=ROOT, prev_root_token_id=PREV,
            order=ORDER)
    want = reference_contexts(child_depth, parents, node_depth, node_token,
                              ROOT, PREV, ORDER)
    check(f"order3 depth{child_depth} parents={parents}", got, want)

print("== order=4 (ctx_len=3) and order=2 (ctx_len=1) ==")
for order in (4, 2):
    for child_depth, parents in [(1, [0]), (2, [1, 3]), (3, [5, 6]), (4, [9])]:
        got = f(child_depth=child_depth, parent_indices=T(parents),
                tokens_t=T(tokens_t), parents_t=T(parents_t),
                root_token_id=ROOT, prev_root_token_id=PREV,
                order=order)
        want = reference_contexts(child_depth, parents, node_depth, node_token,
                                  ROOT, PREV, order)
        check(f"order{order} depth{child_depth} parents={parents}", got, want)

print("== prev=None (no padding token before root) ==")
for child_depth, parents in [(1, [0]), (2, [1]), (3, [5])]:
    got = f(child_depth=child_depth, parent_indices=T(parents),
            tokens_t=T(tokens_t), parents_t=T(parents_t),
            root_token_id=ROOT, prev_root_token_id=None,
            order=3)
    want = reference_contexts(child_depth, parents, node_depth, node_token,
                              ROOT, None, 3)
    check(f"noprev depth{child_depth} parents={parents}", got, want)

print("== candidate-row pairing: ctx[i] must pair with top_ids row i ==")
# Simulate the scoring block: top_ids rows are per-parent (parent order);
# verify that pairing ctx with cand_ids reproduces the probabilities that the
# reference trie would assign to each parent's own candidates given that
# parent's own context.  (Reference pure-Python trie mirroring the C++.)
class _RefTrie:
    def __init__(self, order):
        self.order = order
        self.nodes = [{"freq": 0, "children": {}}]

    def add_conversation(self, toks):
        for start in range(len(toks)):
            seq = toks[start:start + self.order]
            node = self.nodes[0]
            for t in seq:
                node["freq"] += 1
                ch = node["children"]
                if t not in ch:
                    ch[t] = len(self.nodes)
                    self.nodes.append({"freq": 0, "children": {}})
                node = self.nodes[ch[t]]
            node["freq"] += 1

    def get_probability(self, ctx, cands):
        probs = [0.0] * len(cands)
        for length in range(min(self.order - 1, len(ctx)), 0, -1):
            node = self.nodes[0]
            ok = True
            for t in ctx[len(ctx) - length:]:
                ch = node["children"]
                if t not in ch:
                    ok = False
                    break
                node = self.nodes[ch[t]]
            if not ok:
                continue
            cf = node["freq"]
            for i, c in enumerate(cands):
                if probs[i] == 0.0 and c in node["children"]:
                    probs[i] = self.nodes[node["children"][c]]["freq"] / cf
        return probs

corpus = (  # includes the exact 3-grams needed by the probe below
    [ROOT, 13, 22, 31, 13, 22, 31, 13, 22, 99,
     ROOT, 11, 21, 32, 11, 21, 32, 11, 21, 77,
     ROOT, 12, 24, 33, 12, 24, 33, 12, 24, 88,
     ROOT, 13, 22, 31, 11, 21, 32, 12, 24, 33]
)
trie = _RefTrie(ORDER)
trie.add_conversation(corpus)

# parents at depth 3 with their own top-k candidate rows (arbitrary ids);
# each row includes a token that actually follows that parent's context in
# the corpus so the probabilities are non-zero.
parents = [8, 9, 10]
cand_rows = [[13, 31, 999], [11, 32, 21], [12, 33, 31]]  # per-parent rows
ngram_ctx = f(child_depth=4, parent_indices=T(parents),
              tokens_t=T(tokens_t), parents_t=T(parents_t),
              root_token_id=ROOT, prev_root_token_id=PREV,
              order=ORDER)
# expected contexts for parents 8,9,10 = [22,31],[21,32],[24,33]
check("pairing contexts", ngram_ctx, [[22, 31], [21, 32], [24, 33]])
for ctx, cands in zip(ngram_ctx, cand_rows):
    probs = trie.get_probability(ctx, list(cands))
    print(f"    ctx={ctx} cands={cands} -> p={probs}")
# independent per-parent computation (reference contexts keyed by parent)
for parent, cands in zip(parents, cand_rows):
    ref_ctx = reference_contexts(4, [parent], node_depth, node_token,
                                 ROOT, PREV, ORDER)[0]
    got = trie.get_probability(ngram_ctx[parents.index(parent)], list(cands))
    want = trie.get_probability(ref_ctx, list(cands))
    check(f"row-alignment parent{parent} ctx={ref_ctx}", got, want)
    assert any(p > 0.0 for p in want), f"degenerate probe for parent {parent}"
print("    (all probe contexts hit non-zero probabilities)")

print("\nALL EXTENDED NGRAM CONTEXT/PAIRING CHECKS PASSED")
