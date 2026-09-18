"""N-gram interface for DART-style continuity-aware tree scoring.

The tree-expansion scorer in ``eval_dartree.build_dartree_supertree`` consumes
an object exposing :class:`NgramModel`.  The concrete implementation is
:class:`CppTrieNgram`, which wraps DART's C++ ``TrieNgram`` extension
(``utils/ngram_cpp``): it is binary-compatible with DART's ``.trie`` files, so
models published for DART (e.g. ``fvliang/dart-qwen3-ngram`` with
``full.trie`` / ``small.trie``) load as-is.

Loaded models are mutable in memory: :meth:`NgramModel.add_conversation` and
:meth:`NgramModel.add_all` incrementally insert new n-grams into an
already-loaded trie at runtime (extending existing paths and creating new
ones), and :meth:`NgramModel.save` persists the updated table back to DART
``.trie`` format.

When no n-gram model is available the scorer simply receives ``None`` and
skips the n-gram term entirely (see ``eval_dartree.build_dartree_supertree``);
there is no placeholder model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class NgramModel(ABC):
    """DART-style n-gram model used to score candidate token continuations.

    Mirrors the ``TrieNgram::get_probability`` contract from the DART repo:
    given the preceding context tokens and a list of candidate tokens, return
    for each candidate its conditional probability and the length of the
    context suffix that was actually matched (0 = no match / OOV).
    """

    order: int

    @classmethod
    def from_path(cls, path: str) -> "NgramModel":
        """Load an n-gram model from ``path`` (DART ``.trie`` binary format).

        Delegates to :class:`CppTrieNgram`, which wraps DART's C++ extension.
        The extension is JIT-compiled on first use, so this requires torch and
        a C++20 compiler with OpenMP on the host; failures are re-raised with
        a hint.
        """
        return CppTrieNgram.from_path(path)

    @abstractmethod
    def get_probability(
        self,
        context: Sequence[int],
        tokens: Sequence[int],
    ) -> tuple[list[float], list[int]]:
        """Return ``(probs, matched_lengths)`` for every candidate token.

        ``probs[i]`` is ``P(tokens[i] | context)``; OOV candidates score
        ``0.0`` and ``matched_lengths[i]`` is ``0``.  The caller adds a small
        epsilon before taking the log, exactly like DART's
        ``logf(score + 1e-10f)``.
        """
        raise NotImplementedError

    @abstractmethod
    def add_conversation(self, tokens: Sequence[int]) -> None:
        """Incrementally add every n-gram window of ``tokens`` to the model.

        Mirrors ``TrieNgram::add_conversation``: for each start position a
        slice of up to ``order`` tokens is inserted as a sequence, creating
        new trie paths and bumping the frequencies of existing ones.  Safe to
        call at any time on an already-loaded model (runtime update) as long
        as it is not concurrent with :meth:`get_probability` on the same
        object.
        """
        raise NotImplementedError

    @abstractmethod
    def add_all(self, other: "NgramModel") -> None:
        """Merge another model's trie into this one (frequencies summed).

        Equivalent to DART's ``TrieNgram::add_all`` (used to fold partial
        models together); both tries must share the same ``order``.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the current trie to ``path`` in DART ``.trie`` format."""
        raise NotImplementedError


class CppTrieNgram(NgramModel):
    """DART's C++ ``TrieNgram`` behind the :class:`NgramModel` API.

    The underlying pybind11 extension (``utils/ngram_cpp``) is JIT-compiled on
    first use via ``torch.utils.cpp_extension.load`` and cached on disk.  It
    implements a longest-match context walk: for a context of length L it
    scores candidates against the longest suffix (up to ``order - 1`` tokens)
    found in the trie, with raw MLE probabilities ``freq(child) / freq(ctx)``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self.order = int(model.get_order())

    @classmethod
    def from_path(cls, path: str) -> "CppTrieNgram":
        try:
            # Lazy import: the first call triggers the one-time JIT build of
            # the C++ extension (requires torch + a C++20 compiler).
            from utils.ngram_cpp import load_cpp_ngram

            cpp_ngram = load_cpp_ngram()
            model = cpp_ngram.TrieNgram.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load n-gram model from {path!r} via the C++ "
                "TrieNgram extension. Check that the .trie file exists and "
                "that the host has torch and a C++20 compiler with OpenMP "
                "(the extension is JIT-compiled on first use)."
            ) from exc
        return cls(model)

    def get_probability(
        self,
        context: Sequence[int],
        tokens: Sequence[int],
    ) -> tuple[list[float], list[int]]:
        probs, matched_lengths = self._model.get_probability(
            list(context), list(tokens)
        )
        return [float(p) for p in probs], [int(m) for m in matched_lengths]

    def add_conversation(self, tokens: Sequence[int]) -> None:
        self._model.add_conversation(list(tokens))

    def add_all(self, other: "NgramModel") -> None:
        if not isinstance(other, CppTrieNgram):
            raise TypeError(
                "add_all requires another CppTrieNgram-backed model; "
                f"got {type(other).__name__}"
            )
        self._model.add_all(other._model)

    def save(self, path: str) -> None:
        self._model.save(path)
