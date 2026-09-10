"""Unit tests for ``RetrievalAdjacencyMatrix`` (``utils/retrieval.py``).

Covers the GPU-resident top-k successor table used by the Graft retrieval
tree (paper Eq. 9): construction, online updates from target-model logits
(paper Eq. 12/14), mean-pooling over duplicate parents, row overwriting, and
lookups.  All tests run on CPU.

Run from the repository root on a machine with torch (CPU is enough):

    python -m pytest tests/test_retrieval_matrix.py -v
"""

import torch

from utils.retrieval import RetrievalAdjacencyMatrix


def _topk_ids(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Reference top-k selection (descending indices)."""
    return torch.topk(logits, k=k, dim=-1).indices


def test_init_shape_and_fill():
    matrix = RetrievalAdjacencyMatrix(1000, 9, "cpu")
    assert matrix.matrix.shape == (1000, 9)
    assert matrix.k == 9
    assert matrix.vocab_size == 1000
    assert bool((matrix.matrix == -1).all())


def test_update_single_row():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    logits = torch.randn(1, vocab)
    matrix.update_from_logits(torch.tensor([7]), logits)
    expected = _topk_ids(logits, k).squeeze(0)
    assert torch.equal(matrix.matrix[7], expected)


def test_update_multiple_rows():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    tokens = torch.tensor([3, 9, 21])
    logits = torch.randn(3, vocab)
    matrix.update_from_logits(tokens, logits)
    for i, token in enumerate(tokens.tolist()):
        expected = _topk_ids(logits[i : i + 1], k).squeeze(0)
        assert torch.equal(matrix.matrix[token], expected), (
            token,
            matrix.matrix[token],
            expected,
        )


def test_update_duplicate_parents_mean_pool():
    """Same parent seen twice -> M[row] = argtop_k of the *mean* distribution."""
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    tokens = torch.tensor([4, 4, 10])
    logits = torch.randn(3, vocab)
    matrix.update_from_logits(tokens, logits)
    mean = (logits[0] + logits[1]) / 2.0
    expected = _topk_ids(mean.unsqueeze(0), k).squeeze(0)
    assert torch.equal(matrix.matrix[4], expected)


def test_update_overwrites_previous_row():
    """A second update replaces the row entirely (no accumulation)."""
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    matrix.update_from_logits(torch.tensor([4]), torch.randn(1, vocab))
    second = torch.randn(1, vocab)
    matrix.update_from_logits(torch.tensor([4]), second)
    expected = _topk_ids(second, k).squeeze(0)
    assert torch.equal(matrix.matrix[4], expected)


def test_uninitialized_rows_stay_fill():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    matrix.update_from_logits(torch.tensor([0, 1]), torch.randn(2, vocab))
    assert bool((matrix.matrix[2:] == -1).all())


def test_update_empty_is_noop():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    matrix.update_from_logits(
        torch.empty((0,), dtype=torch.long), torch.empty((0, vocab))
    )
    assert bool((matrix.matrix == -1).all())


def test_lookup_returns_stored_values():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    tokens = torch.tensor([7, 8])
    matrix.update_from_logits(tokens, torch.randn(2, vocab))
    ranks = torch.tensor([0, 3])
    out = matrix.lookup(tokens, ranks)
    assert out.dtype == torch.long
    assert out[0].item() == matrix.matrix[7, 0].item()
    assert out[1].item() == matrix.matrix[8, 3].item()


def test_lookup_uninitialized_returns_fill():
    vocab, k = 50, 5
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    out = matrix.lookup(torch.tensor([42]), torch.tensor([0]))
    assert out[0].item() == -1


def test_int32_dtype_supported():
    """The table only stores token ids, so int32 storage is valid end-to-end."""
    matrix = RetrievalAdjacencyMatrix(100, 9, "cpu", dtype=torch.int32)
    assert matrix.matrix.dtype == torch.int32
    matrix.update_from_logits(torch.tensor([1]), torch.randn(1, 100))
    assert matrix.matrix.dtype == torch.int32
    assert matrix.matrix[1].min().item() >= 0
    out = matrix.lookup(torch.tensor([1]), torch.tensor([0]))
    assert out.item() >= 0


def test_update_deterministic():
    """Same inputs -> identical table (reproducibility under a fixed seed)."""
    vocab, k = 50, 5

    torch.manual_seed(0)
    tokens_a = torch.randint(0, vocab, (16,))
    logits_a = torch.randn(16, vocab)

    torch.manual_seed(0)
    tokens_b = torch.randint(0, vocab, (16,))
    logits_b = torch.randn(16, vocab)

    assert torch.equal(tokens_a, tokens_b)
    assert torch.equal(logits_a, logits_b)

    m_a = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    m_b = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    m_a.update_from_logits(tokens_a, logits_a)
    m_b.update_from_logits(tokens_b, logits_b)
    assert torch.equal(m_a.matrix, m_b.matrix)
