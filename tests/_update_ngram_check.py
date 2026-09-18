"""Standalone check for the online n-gram adaptation hook (--update-ngram).

Static checks (no torch needed):
  * eval_dartree.py defines --update-ngram and forwards args.update_ngram
    into dartree_generate(update_ngram=...);
  * dartree_generate accepts update_ngram and guards the in-memory
    add_conversation call with `update_ngram and ngram_model is not None`;
  * run_dartree.py defines --update-ngram and forwards it.

Dynamic check: extract the hook block via AST and execute it against fakes
to verify the exact token segment fed to the trie (lookback token + accepted
chain + next token), and that the guard skips the update when disabled or
when no model is loaded.

Run:  python tests/_update_ngram_check.py
"""

from __future__ import annotations

import ast

EVAL_SRC = open("eval_dartree.py", encoding="utf-8").read()
RUN_SRC = open("run_dartree.py", encoding="utf-8").read()

eval_tree = ast.parse(EVAL_SRC)


def has_arg(n: ast.FunctionDef, name: str) -> bool:
    args = n.args.posonlyargs + n.args.args + n.args.kwonlyargs
    return any(a.arg == name for a in args)


# ---------------------------------------------------------------------------
# Static checks: eval_dartree.py
# ---------------------------------------------------------------------------

# --update-ngram argparse flag must be defined
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "add_argument"
    and any(
        isinstance(a, ast.Constant) and a.value == "--update-ngram"
        for a in n.args
    )
    for n in ast.walk(eval_tree)
), "eval_dartree.py must define --update-ngram"

# dartree_generate must accept update_ngram
gen_fn = next(
    n for n in ast.walk(eval_tree)
    if isinstance(n, ast.FunctionDef) and n.name == "dartree_generate"
)
assert has_arg(gen_fn, "update_ngram"), (
    "dartree_generate must accept update_ngram"
)

# main() must forward args.update_ngram into dartree_generate(...)
main_fn = next(
    n for n in ast.walk(eval_tree)
    if isinstance(n, ast.FunctionDef) and n.name == "main"
)
assert any(
    isinstance(n, ast.keyword)
    and n.arg == "update_ngram"
    and isinstance(n.value, ast.Attribute)
    and n.value.attr == "update_ngram"
    for n in ast.walk(main_fn)
), "main() must pass update_ngram=args.update_ngram to dartree_generate"
print("OK  eval_dartree.py defines + wires --update-ngram")

# ---------------------------------------------------------------------------
# Static checks: run_dartree.py
# ---------------------------------------------------------------------------

run_tree = ast.parse(RUN_SRC)
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "add_argument"
    and any(
        isinstance(a, ast.Constant) and a.value == "--update-ngram"
        for a in n.args
    )
    for n in ast.walk(run_tree)
), "run_dartree.py must define --update-ngram"
assert 'engine_args.append("--update-ngram")' in RUN_SRC, (
    "run_dartree.py must forward --update-ngram"
)
print("OK  run_dartree.py defines + forwards --update-ngram")

# ---------------------------------------------------------------------------
# Dynamic check: extract the hook block and run it against fakes
# ---------------------------------------------------------------------------

hook = next(
    n for n in ast.walk(gen_fn)
    if isinstance(n, ast.If)
    and any(isinstance(t, ast.Name) and t.id == "update_ngram"
            for t in ast.walk(n.test))
)
# the guard must also require a loaded model
assert any(
    isinstance(t, ast.Name) and t.id == "ngram_model"
    for t in ast.walk(hook.test)
), "hook guard must reference ngram_model"
# the hook body must call ngram_model.add_conversation
assert any(
    isinstance(s, ast.Expr)
    and isinstance(s.value, ast.Call)
    and isinstance(s.value.func, ast.Attribute)
    and s.value.func.attr == "add_conversation"
    for s in hook.body
), "hook body must call add_conversation"


class FakeTensor:
    """Stand-in for the accepted_tokens tensor (2-D -> reshape(-1) etc.)."""

    def reshape(self, *args):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [11, 12, 13]


class FakeNgram:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def add_conversation(self, tokens) -> None:
        self.calls.append(list(tokens))


def run_hook(**overrides) -> FakeNgram:
    fake = FakeNgram()
    ns = {
        "update_ngram": True,
        "ngram_model": fake,
        "prev_round_root_token_id": 10,
        "accepted_tokens": FakeTensor(),
        "next_token": 14,
    }
    ns.update(overrides)
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "<hook>", "exec"), ns)
    return fake


# enabled + model loaded -> exactly one add_conversation with
# [lookback, *accepted_chain, next_token]
fake = run_hook()
assert fake.calls == [[10, 11, 12, 13, 14]], fake.calls
print("OK  hook feeds [lookback, accepted chain..., next token] to the trie")

# disabled -> no update
fake = run_hook(update_ngram=False)
assert fake.calls == [], fake.calls
print("OK  update_ngram=False skips the update")

# enabled but no model -> no update
fake = run_hook(ngram_model=None)
assert fake.calls == [], fake.calls
print("OK  ngram_model=None skips the update")

print("\nALL UPDATE-NGRAM CHECKS PASSED")
