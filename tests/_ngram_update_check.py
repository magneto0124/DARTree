"""Torch-free smoke check for the runtime ngram update methods.

Verifies the plumbing of NgramModel.add_conversation / add_all / save with a
duck-typed fake standing in for the C++ TrieNgram (the real extension needs
torch + a C++20 compiler).  Run:  python tests/_ngram_update_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load utils/ngram.py directly: importing the `utils` package would pull in
# torch via utils/__init__.py, which this torch-free check must avoid.
_spec = importlib.util.spec_from_file_location(
    "_dartree_ngram_mod",
    Path(__file__).resolve().parents[1] / "utils" / "ngram.py",
)
_ngram_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ngram_mod)

NgramModel = _ngram_mod.NgramModel
CppTrieNgram = _ngram_mod.CppTrieNgram


class FakeModel:
    """Minimal duck-typed stand-in for the pybind TrieNgram object."""

    def __init__(self, order: int = 3) -> None:
        self.order = order
        self.added: list[list[int]] = []
        self.merged: list[object] = []
        self.saved: list[str] = []

    def get_order(self) -> int:
        return self.order

    def add_conversation(self, tokens) -> None:
        self.added.append(list(tokens))

    def add_all(self, other) -> None:
        self.merged.append(other)

    def save(self, path: str) -> None:
        self.saved.append(path)

    def get_probability(self, context, tokens):
        return ([0.0] * len(tokens)), ([0] * len(tokens))


def main() -> None:
    # 1) ABC contract: all four methods are abstract, so NgramModel() fails.
    assert NgramModel.__abstractmethods__ == {
        "get_probability",
        "add_conversation",
        "add_all",
        "save",
    }, NgramModel.__abstractmethods__
    try:
        NgramModel()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError("NgramModel() should be abstract")

    # 2) CppTrieNgram forwards add_conversation / add_all / save to the C++
    #    backing object (list() conversion included).
    backing = FakeModel(order=3)
    wrapped = CppTrieNgram(backing)
    assert wrapped.order == 3

    wrapped.add_conversation((7, 8, 9))  # tuple -> must be converted to list
    assert backing.added == [[7, 8, 9]], backing.added

    other_backing = FakeModel(order=3)
    other = CppTrieNgram(other_backing)
    wrapped.add_all(other)
    assert backing.merged == [other_backing], backing.merged

    wrapped.save("out.trie")
    assert backing.saved == ["out.trie"], backing.saved

    # 3) add_all rejects objects without a C++ backing.
    try:
        wrapped.add_all(object())
    except TypeError:
        pass
    else:
        raise AssertionError("add_all(object()) should raise TypeError")

    # 4) from_path still resolves to a concrete CppTrieNgram subclass.
    assert CppTrieNgram.from_path.__func__ is not None

    print("OK  ngram update methods (wrapper plumbing) verified")


if __name__ == "__main__":
    main()
