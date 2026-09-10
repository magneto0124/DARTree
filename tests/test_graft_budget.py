"""Tests for Graft Phase 1: fixed-ratio budget split (resolve_graft_retain)."""
from __future__ import annotations

import pytest

from utils.retrieval import resolve_graft_retain


def test_sum_invariant_holds():
    for budget in (1, 8, 64, 100):
        for ratio in (0.1, 0.5, 0.6, 1.0):
            draft_retain, k_ret = resolve_graft_retain(budget, ratio)
            assert draft_retain + k_ret == budget
            assert 1 <= draft_retain <= budget
            assert 0 <= k_ret < budget


def test_ratio_one_degenerates_to_no_retrieval():
    draft_retain, k_ret = resolve_graft_retain(64, 1.0)
    assert draft_retain == 64
    assert k_ret == 0


def test_rounding_sixty_percent():
    draft_retain, k_ret = resolve_graft_retain(64, 0.6)
    assert draft_retain == 38
    assert k_ret == 26


def test_supertree_clamp():
    # Super tree only grew 10 nodes but 0.6*64=38 were requested.
    draft_retain, k_ret = resolve_graft_retain(64, 0.6, supertree_node_count=10)
    assert draft_retain == 10
    assert k_ret == 54
    assert draft_retain + k_ret == 64


def test_supertree_clamp_not_below_one():
    draft_retain, k_ret = resolve_graft_retain(64, 0.6, supertree_node_count=0)
    assert draft_retain == 1
    assert k_ret == 63


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        resolve_graft_retain(0, 0.6)
    with pytest.raises(ValueError):
        resolve_graft_retain(64, 0.0)
    with pytest.raises(ValueError):
        resolve_graft_retain(64, 1.5)
    with pytest.raises(ValueError):
        resolve_graft_retain(-5, 0.6)