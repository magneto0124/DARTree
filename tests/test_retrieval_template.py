"""Unit tests for retrieval-template construction and materialisation.

Covers ``make_decaying_depth_counts``, ``build_rank_template`` and
``RetrievalTemplate.materialize`` (``utils/retrieval.py``) -- the pieces that
build the Graft retrieval tree from the adjacency matrix (paper Sec. 3.2 /
Appendix A.1): the imbalanced template over successor ranks, per-depth
batched gathers, and prefix-closed dropping of nodes whose parent row is
still empty (cold-start degradation).

Run from the repository root on a machine with torch (CPU is enough):

    python -m pytest tests/test_retrieval_template.py -v
"""

import torch
import pytest

from utils.retrieval import (
    RetrievalAdjacencyMatrix,
    RetrievalTemplate,
    build_rank_template,
    make_decaying_depth_counts,
)


# --- make_decaying_depth_counts -------------------------------------------


@pytest.mark.parametrize(
    "total,depth_limit",
    [(52, 9), (36, 9), (20, 9), (15, 15), (64, 15), (1, 9), (7, 3), (12, 4)],
)
def test_decaying_counts_exact_sum(total, depth_limit):
    counts = make_decaying_depth_counts(total, depth_limit)
    assert len(counts) == depth_limit
    assert sum(counts) == total


@pytest.mark.parametrize("total,depth_limit", [(52, 9), (64, 15), (30, 5)])
def test_decaying_counts_monotonic_non_increasing(total, depth_limit):
    counts = make_decaying_depth_counts(total, depth_limit)
    assert counts == sorted(counts, reverse=True)


def test_decaying_counts_min_width():
    counts = make_decaying_depth_counts(30, 5, min_width=2)
    assert sum(counts) == 30
    assert all(count >= 2 for count in counts)


def test_decaying_counts_zero_budget():
    assert make_decaying_depth_counts(0, 9) == [0] * 9


def test_decaying_counts_small_total():
    counts = make_decaying_depth_counts(3, 9, min_width=1)
    assert sum(counts) == 3
    assert all(count >= 0 for count in counts)


# --- build_rank_template ---------------------------------------------------


def test_rank_template_greedy_chain():
    """j == 0 at every depth extends the greedy top-1 chain."""
    parents, ranks, effective = build_rank_template([4, 3, 2], k=5)
    assert effective == [4, 3, 2]
    assert len(parents) == 9
    # depth-1 chain node: template node 1 = (root, rank 0)
    assert parents[0] == 0 and ranks[0] == 0
    # depth-2 chain node: template node 5 = (node 1, rank 0)
    assert parents[4] == 1 and ranks[4] == 0
    # depth-3 chain node: template node 8 = (node 5, rank 0)
    assert parents[7] == 5 and ranks[7] == 0


def test_rank_template_round_robin():
    """Remaining nodes spread round-robin over the previous frontier."""
    parents, ranks, _ = build_rank_template([4, 3], k=5)
    # depth 1: nodes 1..4, parents = root, ranks 0..3
    for j in range(4):
        assert parents[j] == 0
        assert ranks[j] == j
    # depth 2: nodes 5..7 -> j in 1..3 -> parent = frontier[(j-1) % 4],
    # rank = (j-1) // 4 + 1
    frontier = list(range(1, 5))
    for j in range(1, 4):
        node = 4 + j
        assert parents[node - 1] == frontier[(j - 1) % 4]
        assert ranks[node - 1] == (j - 1) // 4 + 1


def test_rank_template_width_clamped_by_k():
    """Width is clamped so no successor rank exceeds k-1."""
    parents, ranks, effective = build_rank_template([10, 10, 10], k=4)
    # depth 1: frontier_len=1 -> width = min(10, 1 + 1*3) = 4
    assert effective[0] == 4
    # deeper depths: frontier_len=4 -> 1 + 4*3 = 13 >= 10, no clamping
    assert effective[1] == 10
    assert effective[2] == 10
    assert all(0 <= rank < 4 for rank in ranks)


def test_rank_template_prefix_closed_parents():
    parents, ranks, _ = build_rank_template([8, 16, 14], k=9)
    assert len(parents) == 8 + 16 + 14
    for node_index, parent in enumerate(parents, start=1):
        assert 0 <= parent < node_index, (node_index, parent)


# --- RetrievalTemplate -----------------------------------------------------


def test_template_metadata():
    template = RetrievalTemplate([4, 3, 2], k=5)
    assert template.node_count == 9
    assert template.max_depth == 3
    assert template.depth_counts == [4, 3, 2]
    assert template.node_depths == [1, 1, 1, 1, 2, 2, 2, 3, 3]


def _fill_matrix(matrix, rows):
    """Fill rows via one-hot-ish logits so argtop_k == the given successor ids."""
    vocab = matrix.vocab_size
    for token, successors in rows.items():
        logits = torch.zeros(1, vocab)
        for rank, successor in enumerate(successors):
            logits[0, successor] = float(matrix.k - rank)
        matrix.update_from_logits(torch.tensor([token]), logits)


def test_materialize_full_matrix():
    """With every consulted row filled, all nodes materialise and equal the
    matrix lookups M[token(parent), rank] (paper Eq. 10/15)."""
    vocab, k = 100, 3
    template = RetrievalTemplate([2, 2], k)
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    _fill_matrix(matrix, {5: [10, 11, 12], 10: [20, 21, 22]})

    tokens, parents, ranks, valid = template.materialize(5, matrix)
    assert valid == [True, True, True, True]
    assert tokens == [10, 11, 20, 21]
    assert parents == [0, 0, 1, 1]
    assert ranks == [0, 1, 0, 1]


def test_materialize_cold_root_drops_everything():
    """Root row still empty -> every node drops (cold-start degradation)."""
    vocab, k = 100, 3
    template = RetrievalTemplate([2, 2], k)
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    tokens, parents, ranks, valid = template.materialize(5, matrix)
    assert valid == [False, False, False, False]
    assert tokens == [-1, -1, -1, -1]


def test_materialize_partial_is_prefix_closed():
    """Nodes whose parent row is empty are dropped together with their whole
    subtree; surviving nodes keep their template parents."""
    vocab, k = 100, 3
    template = RetrievalTemplate([2, 2, 2, 2], k)
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    # root row and the row of template node 1 (token 10) are filled; the row
    # of node 2 (token 11) is not.
    _fill_matrix(matrix, {5: [10, 11, 12], 10: [20, 21, 22]})

    tokens, parents, ranks, valid = template.materialize(5, matrix)
    # depth1: nodes 1, 2 valid. depth2: nodes 3, 4 (parents 1) valid.
    # depth3: nodes 5, 6 (parent 3, token 20) -> row 20 empty -> invalid.
    # depth4: nodes 7, 8 (parent 5) -> invalid (inherited).
    assert valid == [True, True, True, True, False, False, False, False]
    assert tokens[:4] == [10, 11, 20, 21]
    assert tokens[4:] == [-1, -1, -1, -1]
    # prefix closure: a node whose parent failed is dropped too (inherited
    # invalidity propagates down the subtree)
    for node_index, (parent, is_valid) in enumerate(zip(parents, valid), start=1):
        if parent > 0 and not valid[parent - 1]:
            assert not is_valid, (node_index, parent)


def test_materialize_consistent_with_lookup():
    """For a fully-filled matrix, every materialised token equals a batched
    lookup of the same (parent token, rank) -- i.e. the retrieval tree is
    exactly a walk over the adjacency matrix."""
    vocab, k = 300, 5
    counts = make_decaying_depth_counts(20, 5, min_width=1)
    template = RetrievalTemplate(counts, k)
    root = 11
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    torch.manual_seed(0)
    for token in range(vocab):
        matrix.update_from_logits(
            torch.tensor([token]), torch.randn(1, vocab)
        )

    tokens, parents, ranks, valid = template.materialize(root, matrix)
    assert all(valid)
    node_tokens = [root]
    for index in range(len(tokens)):
        parent_token = node_tokens[parents[index]]
        expected = matrix.lookup(
            torch.tensor([parent_token]), torch.tensor([ranks[index]])
        ).item()
        assert tokens[index] == expected, (index, tokens[index], expected)
        node_tokens.append(tokens[index])


def test_materialize_root_accepts_tensor():
    vocab, k = 100, 3
    template = RetrievalTemplate([1], k)
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    _fill_matrix(matrix, {5: [10, 11, 12]})
    tokens, parents, ranks, valid = template.materialize(
        torch.tensor(5, dtype=torch.long), matrix
    )
    assert valid == [True]
    assert tokens == [10]
