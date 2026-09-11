"""Tests for the Graft retrieval primitives (utils.retrieval).

Pure-torch, CPU-runnable; no model weights required.
"""
from __future__ import annotations

import pytest
import torch

from utils.retrieval import (
    GraftAdjacencyMatrix,
    build_retrieval_template,
    default_level_widths,
)


def test_update_lookup_roundtrip():
    m = GraftAdjacencyMatrix(vocab_size=16, k=3, device="cpu", pad_token_id=0)
    # Token 5 -> successors [7, 2, 9]; token 8 -> [1, 4, 11].  All logits are
    # explicitly ranked (no ties): torch.topk's tie-breaking is NOT guaranteed
    # to pick the smallest index, so tied rows would make this flaky.
    ids = torch.tensor([5, 8])
    logits = torch.zeros(2, 16)
    logits[0, [7, 2, 9]] = torch.tensor([3.0, 2.0, 1.0])
    logits[1, [1, 4, 11]] = torch.tensor([3.0, 2.0, 1.0])
    m.update(ids, logits)

    assert m.matrix[5].tolist() == [7, 2, 9]
    assert m.matrix[8].tolist() == [1, 4, 11]
    assert m.lookup(torch.tensor([5, 8]), torch.tensor([0, 2])).tolist() == [7, 11]
    assert m.is_ready(torch.tensor([5, 8, 3])).tolist() == [True, True, False]
    assert m.ready_rows() == 2


def test_update_skips_out_of_range_ids():
    m = GraftAdjacencyMatrix(vocab_size=8, k=2, device="cpu", pad_token_id=99)
    ids = torch.tensor([-1, 0, 8, 3])
    logits = torch.zeros(4, 8)
    logits[1, 1] = 1.0
    logits[3, 5] = 1.0
    m.update(ids, logits)
    assert m.ready_rows() == 2
    assert m.initialized[0].item() and m.initialized[3].item()
    assert not m.initialized[7].item()
    # Out-of-range id 8 would have indexed the last row if not filtered.
    assert m.initialized[7].item() is False


def test_uninitialized_rows_return_pad_and_flag_false():
    m = GraftAdjacencyMatrix(vocab_size=6, k=2, device="cpu", pad_token_id=42)
    got = m.lookup(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0]))
    assert got.tolist() == [42, 42, 42]
    assert m.is_ready(torch.tensor([0, 1, 2])).sum().item() == 0


def test_lookup_broadcast_per_node():
    m = GraftAdjacencyMatrix(vocab_size=8, k=3, device="cpu")
    ids = torch.arange(8)
    logits = torch.eye(8) * 10
    m.update(ids, logits)
    # Per-node lookup: one (parent, rank) pair per element.
    parents = torch.tensor([0, 1, 2, 3])
    ranks = torch.tensor([0, 1, 2, 2])
    assert m.lookup(parents, ranks).shape == (4,)
    # Scalar-rank broadcast across many parents.
    assert m.lookup(parents, 0).shape == (4,)


def test_lookup_rank_out_of_range_raises():
    m = GraftAdjacencyMatrix(vocab_size=4, k=2, device="cpu")
    with pytest.raises(IndexError):
        m.lookup(torch.tensor([0]), torch.tensor([2]))
    with pytest.raises(IndexError):
        m.lookup(torch.tensor([0]), 5)


def test_state_dict_roundtrip():
    m = GraftAdjacencyMatrix(vocab_size=4, k=2, device="cpu")
    m.update(torch.tensor([1]), torch.tensor([[0.0, 5.0, 4.0, 3.0]]))
    m2 = GraftAdjacencyMatrix(vocab_size=4, k=2, device="cpu")
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.matrix, m.matrix)
    assert torch.equal(m2.initialized, m.initialized)


def test_build_retrieval_template_root_children_and_chain():
    widths = [3, 2, 1]
    parents, ranks, depths = build_retrieval_template(widths)
    # 6 nodes total.
    assert len(parents) == len(ranks) == len(depths) == 6
    # Depth 1: three root children with ranks 0,1,2.
    assert parents[:3] == [-1, -1, -1]
    assert ranks[:3] == [0, 1, 2]
    assert depths[:3] == [1, 1, 1]
    # Prefix-closed: parent index < child index.
    for child, parent in enumerate(parents):
        if parent != -1:
            assert parent < child
    # Depth-2 nodes: frontier is [0,1,2] -> parents round-robin [0,1].
    assert parents[3:5] == [0, 1]
    assert ranks[3:5] == [0, 0]
    assert depths[3:5] == [2, 2]


def test_build_retrieval_template_rank0_greedy_chain():
    widths = [1, 1, 1, 1]
    parents, ranks, depths = build_retrieval_template(widths)
    assert parents == [-1, 0, 1, 2]
    assert ranks == [0, 0, 0, 0]
    assert depths == [1, 2, 3, 4]


def test_build_retrieval_template_no_frontier_stops():
    # A zero-width deeper level after layer one terminates cleanly.
    parents, ranks, depths = build_retrieval_template([2, 0, 3])
    assert depths == [1, 1]


def test_default_level_widths_budgets():
    assert default_level_widths(0, 4) == []
    assert default_level_widths(1, 4) == [1]
    assert sum(default_level_widths(10, 4)) == 10
    w = default_level_widths(28, 4, root_width=8)
    assert w[0] == 8 and len(w) <= 4 and sum(w) == 28
    # Every level keeps at least one node (root width is capped).
    assert all(x >= 1 for x in default_level_widths(3, 4, root_width=100))


def test_default_level_widths_small_budget_keeps_root_breadth():
    # A small budget with a large depth limit must NOT degenerate into a
    # single rank-0 chain: the root layer keeps its breadth first.
    w = default_level_widths(8, 16)
    assert w[0] >= 4  # root layer holds at least half the budget
    assert sum(w) == 8
    assert len(w) <= 16
    # Deeper levels still keep at least one node each.
    assert all(x >= 1 for x in w)
    # max_depth=1 keeps everything on a single level.
    assert default_level_widths(8, 1) == [8]