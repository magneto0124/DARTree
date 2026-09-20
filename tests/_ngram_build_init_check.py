"""Standalone check for ngram_build.py's update mode (--init-ngram).

Static checks (no torch / tqdm needed):
  * --init-ngram is a defined, optional argparse flag;
  * workers (build_ngram_partition) still build fresh partials -- they must
    NOT load the base model (the base is folded in exactly once at merge);
  * merge_partial_models accepts init_path: with it, the base model seeds the
    merge and EVERY partial is folded on top; without it, the first partial
    seeds the merge (build-from-scratch);
  * main() validates that the base model's order matches --ngram-order and
    forwards init_path=args.init_ngram to the merge.

Dynamic checks:
  * read_trie_order reads only the 8-byte little-endian order header;
  * merge control flow against fakes: update mode loads exactly [base,
    *partials] and folds all partials; from-scratch mode loads
    [partial_0, *partial_rest] and folds the rest.

Run:  python tests/_ngram_build_init_check.py
"""

from __future__ import annotations

import ast
import os
import struct
import tempfile

SRC = open("utils/ngram_build.py", encoding="utf-8").read()
tree = ast.parse(SRC)


def fn(name: str) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

# --init-ngram must be a defined, optional argparse flag
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "add_argument"
    and any(
        isinstance(a, ast.Constant) and a.value == "--init-ngram"
        for a in n.args
    )
    for n in ast.walk(tree)
), "ngram_build.py must define --init-ngram"
assert "default=None" in SRC, "--init-ngram must be optional (default=None)"
print("OK  --init-ngram defined as an optional flag")

# workers must stay fresh: build_ngram_partition must not touch init_ngram
worker_fn = fn("build_ngram_partition")
worker_refs = {
    n.id for n in ast.walk(worker_fn) if isinstance(n, ast.Name)
} | {
    n.attr for n in ast.walk(worker_fn) if isinstance(n, ast.Attribute)
}
assert "init_ngram" not in worker_refs, (
    "workers must not load the base model (it would be counted n_jobs times)"
)
print("OK  workers build fresh partials (base folded once at merge)")

# merge_partial_models must accept init_path
merge_fn = fn("merge_partial_models")
args_ = merge_fn.args.posonlyargs + merge_fn.args.args + merge_fn.args.kwonlyargs
assert any(a.arg == "init_path" for a in args_), (
    "merge_partial_models must accept init_path"
)
assert any(
    isinstance(d, ast.Constant) and d.value is None
    for d in merge_fn.args.defaults
), "init_path must default to None"
assert "TrieNgram.load(init_path)" in SRC, (
    "update mode must load the base model at init_path"
)
assert "to_add = list(partial_files)" in SRC, (
    "update mode must fold EVERY partial on top of the base"
)
assert "TrieNgram.load(partial_files[0])" in SRC, (
    "from-scratch mode must seed the merge with the first partial"
)
print("OK  merge logic: update mode = base + all partials; scratch = first partial + rest")

# main() must validate order and forward init_path
main_fn = fn("main")
assert any(
    isinstance(n, ast.Raise)
    and isinstance(n.exc, ast.Call)
    and isinstance(n.exc.func, ast.Name)
    and n.exc.func.id == "ValueError"
    for n in ast.walk(main_fn)
), "main() must raise ValueError on order mismatch"
assert any(
    isinstance(n, ast.keyword)
    and n.arg == "init_path"
    and isinstance(n.value, ast.Attribute)
    and n.value.attr == "init_ngram"
    for n in ast.walk(main_fn)
), "main() must pass init_path=args.init_ngram to the merge"
print("OK  main() validates order and forwards init_path")

# ---------------------------------------------------------------------------
# Dynamic checks
# ---------------------------------------------------------------------------

# read_trie_order: only the 8-byte order header is read
read_fn = fn("read_trie_order")
ns: dict = {"struct": struct}
exec(compile(ast.Module(body=[read_fn], type_ignores=[]), "<read>", "exec"), ns)
read_trie_order = ns["read_trie_order"]

with tempfile.TemporaryDirectory() as tmp:
    header = os.path.join(tmp, "header.trie")
    with open(header, "wb") as f:
        f.write(struct.pack("<Q", 5))          # order = 5
        f.write(struct.pack("<Q", 123456))     # node_count (ignored)
        f.write(b"\x00" * 64)                  # trailing junk (ignored)
    assert read_trie_order(header) == 5, read_trie_order(header)
    print("OK  read_trie_order reads the 8-byte order header only")

    # merge control flow against fakes
    class _FakeMerged:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.added: list[str] = []
            self.saved_path: str | None = None

        def add_all(self, other) -> None:
            self.added.append(other.tag)

        def save(self, path: str) -> None:
            self.saved_path = path

    class _FakeCppNgram:
        loaded: list[str] = []
        merged_obj: "_FakeMerged | None" = None

        class TrieNgram:
            @staticmethod
            def load(path: str) -> "_FakeMerged":
                _FakeCppNgram.loaded.append(path)
                obj = _FakeMerged(path)
                if _FakeCppNgram.merged_obj is None:
                    _FakeCppNgram.merged_obj = obj
                return obj

    def fake_load_cpp_ngram():
        return _FakeCppNgram

    def fake_tqdm(iterable, **kwargs):
        return iterable

    from types import SimpleNamespace

    merge_ns: dict = {
        "load_cpp_ngram": fake_load_cpp_ngram,
        "tqdm": fake_tqdm,
        "os": os,
        "argparse": __import__("argparse"),
        "Optional": __import__("typing").Optional,
        "Sequence": __import__("typing").Sequence,
    }
    # drop the in-function `from utils.ngram_cpp import load_cpp_ngram` import,
    # keep the def wrapper (the early-exit `return` needs a function scope)
    merge_fn.body = [
        n for n in merge_fn.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    exec(
        compile(ast.Module(body=[merge_fn], type_ignores=[]), "<merge>", "exec"),
        merge_ns,
    )
    merge_partial_models = merge_ns["merge_partial_models"]

    part_a = os.path.join(tmp, "3gram-part0.trie")
    part_b = os.path.join(tmp, "3gram-part1.trie")
    part_c = os.path.join(tmp, "3gram-part2.trie")
    base = os.path.join(tmp, "base.trie")
    for p in (part_a, part_b, part_c, base):
        open(p, "wb").write(b"fake")

    args = SimpleNamespace(output_path=tmp, ngram_order=3)
    final_output = os.path.join(tmp, "3gram.trie")

    # update mode: base seeds the merge, every partial is folded on top
    _FakeCppNgram.loaded = []
    _FakeCppNgram.merged_obj = None
    merge_partial_models(args, [part_a, part_b, part_c], init_path=base)
    assert _FakeCppNgram.loaded == [base, part_a, part_b, part_c], (
        _FakeCppNgram.loaded
    )
    assert _FakeCppNgram.merged_obj is not None
    assert _FakeCppNgram.merged_obj.added == [part_a, part_b, part_c], (
        _FakeCppNgram.merged_obj.added
    )
    assert _FakeCppNgram.merged_obj.saved_path == final_output
    assert not os.path.exists(part_a) and not os.path.exists(part_b)
    print("OK  update mode: base + all partials folded, saved, partials removed")

    # from-scratch mode: first partial seeds the merge, the rest are folded
    # (recreate the partials: the update-mode run above removed them)
    for p in (part_a, part_b, part_c):
        open(p, "wb").write(b"fake")
    _FakeCppNgram.loaded = []
    _FakeCppNgram.merged_obj = None
    merge_partial_models(args, [part_a, part_b, part_c], init_path=None)
    assert _FakeCppNgram.loaded == [part_a, part_b, part_c], (
        _FakeCppNgram.loaded
    )
    assert _FakeCppNgram.merged_obj is not None
    assert _FakeCppNgram.merged_obj.added == [part_b, part_c], (
        _FakeCppNgram.merged_obj.added
    )
    assert _FakeCppNgram.merged_obj.saved_path == final_output
    print("OK  from-scratch mode: first partial seeds, rest folded, saved")

print("\nALL NGRAM-BUILD INIT CHECKS PASSED")
