"""N-gram interface for DART-style continuity-aware tree scoring.

The tree-expansion scorer in ``eval_dartree.build_dartree_supertree`` consumes
an object exposing :class:`NgramModel`.  The concrete 2-gram table is filled
in later; :class:`NoopNgram` is the placeholder so the DART-style scoring path
can be wired up and exercised end-to-end right now (it contributes an
all-zero n-gram term, so selection is unaffected).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class NgramModel(ABC):
    """DART-style n-gram model used to score candidate token continuations.

    Mirrors the ``TrieNgram::get_probability`` contract from the DART repo:
    given the preceding context tokens and a list of candidate tokens, return
    for each candidate its conditional probability and the length of the
    context suffix that was actually matched (0 = no match / OOV).
    """

    order: int

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


class NoopNgram(NgramModel):
    """Placeholder contributing nothing (all-zero probabilities).

    Since every candidate of a parent receives the same constant score
    offset, the ranking (and therefore the selected tree) is unchanged; the
    logit-only score remains the effective criterion.
    """

    order = 2

    def get_probability(
        self,
        context: Sequence[int],
        tokens: Sequence[int],
    ) -> tuple[list[float], list[int]]:
        del context  # unused placeholder
        return [0.0] * len(tokens), [0] * len(tokens)
