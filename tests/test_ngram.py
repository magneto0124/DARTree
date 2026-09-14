"""Tests for the DART-style n-gram model interface (utils/ngram.py).

The pure-Python tests (NoopNgram contract, reference-trie semantics vs an
independent counter-based n-gram model) run anywhere.  Tests exercising the
C++ ``TrieNgram`` extension are skipped when torch / a C++20 compiler is not
available (e.g. on a host without the build toolchain); they should be run on
the target evaluation machine (Linux, torch + gcc).
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter, defaultdict

import pytest

from utils.ngram import NgramModel, NoopNgram


# ============================================================================
# Pure-Python reference mirrors of the C++ TrieNgram semantics
# ============================================================================

class _PyTrie:
    """Minimal pure-Python port of DART's C++ TrieNgram.

    Mirrors ``trie_ngram.cpp`` exactly: ``add_conversation`` adds every
    ``order``-token window, ``get_probability`` walks the longest matching
    context suffix (up to ``order - 1`` tokens) and scores
    ``freq(child) / freq(ctx)`` with 0 for OOV candidates.
    """

    def __init__(self, order: int) -> None:
        self.order = order
        self.nodes: list[dict] = [
            {"token": 0, "parent": 0, "freq": 0, "children": {}}
        ]

    def add_conversation(self, tokens: list[int]) -> None:
        n = len(tokens)
        for start in range(n):
            self._add_sequence(tokens[start : min(n, start + self.order)])

    def _add_sequence(self, seq: list[int]) -> None:
        node = self.nodes[0]
        for tok in seq:
            node["freq"] += 1
            children = node["children"]
            if tok not in children:
                children[tok] = len(self.nodes)
                self.nodes.append(
                    {"token": tok, "parent": 0, "freq": 0, "children": {}}
                )
            node = self.nodes[children[tok]]
        node["freq"] += 1

    def get_probability(
        self, context: list[int], tokens: list[int]
    ) -> tuple[list[float], list[int]]:
        probs = [0.0] * len(tokens)
        matched = [0] * len(tokens)
        for length in range(min(self.order - 1, len(context)), 0, -1):
            start = len(context) - length
            node = self.nodes[0]
            for tok in context[start:]:
                if tok not in node["children"]:
                    node = None
                    break
                node = self.nodes[node["children"][tok]]
            if node is None:
                continue
            ctx_freq = node["freq"]
            for i, tok in enumerate(tokens):
                if matched[i] > 0:
                    continue
                if tok in node["children"]:
                    probs[i] = self.nodes[node["children"][tok]]["freq"] / ctx_freq
                    matched[i] = length
        return probs, matched


def _counter_ngram(corpus: list[int], order: int):
    """Independent n-gram count table used to validate the trie semantics."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for n in range(1, order + 1):
        for i in range(len(corpus) - n + 1):
            counts[n][tuple(corpus[i : i + n])] += 1
    return counts


def _counter_probability(
    counts, context: list[int], tokens: list[int], order: int
) -> tuple[list[float], list[int]]:
    """Longest-match probability from the count table (DART semantics)."""
    probs = [0.0] * len(tokens)
    matched = [0] * len(tokens)
    ctx = context[-(order - 1) :] if context else []
    for i, tok in enumerate(tokens):
        for length in range(len(ctx), 0, -1):
            suffix = tuple(ctx[-length:])
            num = counts[length + 1].get(suffix + (tok,))
            den = counts[length].get(suffix)
            if num is not None and den:
                probs[i] = num / den
                matched[i] = length
                break
    return probs, matched


# ============================================================================
# Pure-Python tests (run everywhere)
# ============================================================================

def test_noop_ngram_contract() -> None:
    model = NoopNgram()
    assert model.order == 2
    probs, matched = model.get_probability([1, 2], [3, 4, 5])
    assert probs == [0.0, 0.0, 0.0]
    assert matched == [0, 0, 0]


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(
    "corpus",
    [
        [1, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2],
        [5, 5, 5, 1, 2, 5, 3, 5, 1, 5],
        [42],
        [],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3],
    ],
)
def test_reference_trie_matches_counter(order: int, corpus: list[int]) -> None:
    """The trie semantics (longest match, freq/freq) match n-gram counts."""
    trie = _PyTrie(order)
    trie.add_conversation(corpus)
    counts = _counter_ngram(corpus, order)

    queries = [
        ([], [1, 2, 3, 4]),
        ([2], [1, 2, 3, 4]),
        ([2, 3], [1, 2, 3, 4]),
        ([9], [1, 2, 3, 4]),
        ([2, 9], [1, 2, 3, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
    ]
    for context, tokens in queries:
        tp, tm = trie.get_probability(context, tokens)
        cp, cm = _counter_probability(counts, context, tokens, order)
        assert tp == pytest.approx(cp, abs=1e-9)
        assert tm == cm


# ============================================================================
# C++ extension tests (skipped when torch / compiler unavailable)
# ============================================================================

def _cpp_ngram():
    pytest.importorskip("torch")
    try:
        from utils.ngram_cpp import load_cpp_ngram

        return load_cpp_ngram()
    except Exception as exc:  # no compiler / build failure
        pytest.skip(f"C++ n-gram extension unavailable: {exc}")


@pytest.mark.parametrize("order", [2, 3])
def test_cpp_parity_with_reference(order: int) -> None:
    """The C++ extension must agree with the reference trie bit-for-bit."""
    cpp = _cpp_ngram()
    corpus = [1, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2, 7, 8, 1, 2, 3, 2, 3, 9]

    ref = _PyTrie(order)
    ref.add_conversation(corpus)
    cpp_model = cpp.TrieNgram(order)
    cpp_model.add_conversation(corpus)
    assert int(cpp_model.get_order()) == order

    for context in ([], [2], [2, 3], [9], [2, 9], [1, 2, 3], [3, 2, 1, 7]):
        for tokens in ([1, 2, 3, 4], [2, 3], [7, 8, 9]):
            rp, rm = ref.get_probability(list(context), list(tokens))
            cp, cm = cpp_model.get_probability(list(context), list(tokens))
            assert [float(p) for p in cp] == pytest.approx(rp, abs=1e-6)
            assert [int(m) for m in cm] == rm


def test_cpp_save_load_roundtrip() -> None:
    cpp = _cpp_ngram()
    model = cpp.TrieNgram(3)
    model.add_conversation([1, 2, 3, 2, 2, 1, 3, 2, 3, 1, 2])

    before = model.get_probability([2, 3], [1, 2, 3])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.trie")
        model.save(path)
        loaded = cpp.TrieNgram.load(path)
        after = loaded.get_probability([2, 3], [1, 2, 3])

    assert [float(p) for p in after[0]] == pytest.approx(
        [float(p) for p in before[0]], abs=1e-6
    )
    assert [int(m) for m in after[1]] == [int(m) for m in before[1]]


def test_cpp_trie_ngram_interface() -> None:
    """CppTrieNgram / NgramModel.from_path must load a saved .trie."""
    pytest.importorskip("torch")
    try:
        from utils.ngram_cpp import load_cpp_ngram
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"C++ n-gram extension unavailable: {exc}")

    from utils.ngram import CppTrieNgram

    cpp = load_cpp_ngram()
    model = cpp.TrieNgram(2)
    model.add_conversation([1, 2, 1, 2, 1, 3, 2])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.trie")
        model.save(path)

        loaded = CppTrieNgram.from_path(path)
        assert isinstance(loaded, NgramModel)
        assert loaded.order == 2

        direct = model.get_probability([2], [1, 2, 3])
        wrapped = loaded.get_probability([2], [1, 2, 3])
        assert wrapped[0] == pytest.approx([float(p) for p in direct[0]], abs=1e-6)
        assert wrapped[1] == [int(m) for m in direct[1]]
