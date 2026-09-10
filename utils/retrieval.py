"""GPU-resident retrieval (Graft) support for DARTree.

Implements the retrieval side of *"Draft Less, Retrieve More: Hybrid Tree
Construction for Speculative Decoding"* (Shen et al., 2026):

* a GPU-resident adjacency matrix ``M`` of shape ``[vocab_size, k]`` whose row
  ``M[t]`` stores the top-``k`` successor tokens of vocabulary token ``t``
  (paper Eq. 9);
* static, unbalanced retrieval templates over successor ranks
  (paper Sec. 3.2 / Appendix A.1) that are materialized from the current root
  token with batched GPU gathers, so the retrieval critical path scales with
  template *depth* instead of node count;
* online updates of ``M`` from target-model verification logits over the whole
  verified tree -- accepted and rejected nodes alike (paper Eq. 12/14).

The pruning side (confidence checkpoints that decide how much draft budget is
released) lives in ``eval_dartree.build_dartree_supertree``; this module only
provides the retrieval machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class RetrievalAdjacencyMatrix:
    """GPU-resident top-k successor table (shape ``[vocab_size, k]``).

    Rows are updated online from target-model next-token distributions: after
    every verification round, ``M[x] = argtop_k(p)`` where ``x`` is a verified
    tree node's token and ``p`` is the target distribution at that node
    (paper Eq. 12/14).  Rows that have never been updated contain ``fill``
    (default ``-1``); template materialization drops nodes whose parent row is
    still empty, so a cold-start matrix degrades gracefully to no retrieval.
    """

    def __init__(
        self,
        vocab_size: int,
        k: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.long,
        fill: int = -1,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.k = max(1, int(k))
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self.matrix = torch.full(
            (self.vocab_size, self.k),
            int(fill),
            dtype=dtype,
            device=self.device,
        )

    def update_from_logits(
        self,
        parent_tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Set ``M[parent] = argtop_k(mean logits over duplicate parents)``.

        ``parent_tokens``: ``[N]`` integer tensor of verified node tokens.
        ``logits``:        ``[N, V]`` float tensor; row ``i`` is the target
                           next-token distribution at node ``i`` (the
                           distribution that predicts successors of
                           ``parent_tokens[i]``).
        """
        if parent_tokens.numel() == 0:
            return
        logits = logits.reshape(-1, logits.shape[-1]).float()
        tokens = parent_tokens.reshape(-1).long()
        k = min(self.k, logits.shape[-1])
        uniq, inverse = torch.unique(tokens, return_inverse=True)
        if uniq.numel() == 0:
            return
        sums = torch.zeros(
            (uniq.numel(), logits.shape[-1]),
            dtype=torch.float32,
            device=logits.device,
        )
        sums.index_add_(0, inverse, logits)
        counts = torch.bincount(inverse, minlength=int(uniq.numel()))
        counts = counts.unsqueeze(1).clamp_min(1)
        means = sums / counts
        top_ids = torch.topk(means, k=k, dim=-1).indices.to(self.dtype)
        self.matrix.index_copy_(0, uniq.to(self.device), top_ids)

    def lookup(
        self,
        parent_tokens: torch.Tensor,
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        """Gather ``M[parent, rank]`` -> child token ids (``[N]``)."""
        parent_tokens = parent_tokens.to(self.device)
        ranks = ranks.to(self.device)
        return self.matrix[parent_tokens, ranks]

    def __repr__(self) -> str:
        return (
            f"RetrievalAdjacencyMatrix(vocab_size={self.vocab_size}, "
            f"k={self.k}, device={self.device})"
        )


def make_decaying_depth_counts(
    total: int,
    depth_limit: int,
    min_width: int = 1,
) -> list[int]:
    """Distribute ``total`` template nodes across ``depth_limit`` depths with a
    decreasing profile (shallow depths receive more nodes than deep ones).

    This mirrors the imbalanced retrieval templates of the paper
    (Appendix A.1 / Table 7): the greedy top-1 continuation chain runs deep,
    while lower-ranked alternatives get fewer descendants.

    Returns a list of per-depth node counts that sums to exactly ``total``
    (each entry at least ``min_width`` when possible).
    """
    depth_limit = max(1, int(depth_limit))
    total = max(0, int(total))
    min_width = max(0, int(min_width))
    if total == 0:
        return [0] * depth_limit
    weights = [1.0 / math.sqrt(float(d) + 1.0) for d in range(depth_limit)]
    weight_sum = sum(weights)
    counts = [
        max(0, int(round(total * w / weight_sum))) for w in weights
    ]
    diff = total - sum(counts)
    index = 0
    while diff > 0:
        counts[index % depth_limit] += 1
        diff -= 1
        index += 1
    depth = depth_limit - 1
    while diff < 0:
        while depth >= 0 and counts[depth] <= min_width:
            depth -= 1
        if depth < 0:
            break
        counts[depth] -= 1
        diff += 1
    return counts


def build_rank_template(
    depth_counts: list[int],
    k: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build a static, unbalanced retrieval template over successor ranks.

    Returns ``(parents, ranks, effective_counts)`` for template nodes
    ``1..N`` (root = 0), where node ``i``'s token is
    ``M[token(parent_i), rank_i]`` (paper Eq. 10/15).

    The all-zero rank path (the greedy top-1 continuation) is extended at
    every depth; the remaining nodes are spread round-robin over the previous
    depth's nodes so higher-ranked parents receive more children and deeper
    extensions.  ``effective_counts`` is ``depth_counts`` clamped so that no
    rank exceeds ``k - 1``.
    """
    k = max(1, int(k))
    parents: list[int] = []
    ranks: list[int] = []
    effective_counts: list[int] = []
    frontier = [0]  # template node ids at the previous depth (0 = root)
    for width in depth_counts:
        width = max(0, int(width))
        frontier_len = len(frontier)
        width = min(width, 1 + frontier_len * (k - 1))
        effective_counts.append(width)
        children: list[int] = []
        for j in range(width):
            if j == 0:
                parent, rank = frontier[0], 0
            else:
                parent = frontier[(j - 1) % frontier_len]
                rank = (j - 1) // frontier_len + 1
            parents.append(parent)
            ranks.append(rank)
            children.append(len(parents))
        frontier = children
    return parents, ranks, effective_counts


class RetrievalTemplate:
    """A static, unbalanced retrieval template over successor ranks.

    The template is rooted at the *current root token* and filled in a
    BFS-like manner: nodes at the same depth are independent once their
    parents are known, so each depth is materialized with one batched GPU
    gather from the adjacency matrix (paper Sec. 3.2).
    """

    def __init__(self, depth_counts: list[int], k: int) -> None:
        self.k = max(1, int(k))
        parents, ranks, effective_counts = build_rank_template(
            depth_counts, self.k
        )
        self.parents = parents          # template-local parent (0 = root)
        self.ranks = ranks              # successor rank
        self.depth_counts = effective_counts  # per-depth node counts
        self.node_count = len(parents)
        self.max_depth = len(self.depth_counts)
        self.node_depths: list[int] = []
        for depth, width in enumerate(self.depth_counts, start=1):
            self.node_depths.extend([depth] * width)

    def materialize(
        self,
        root_token_id: torch.Tensor | int,
        matrix: RetrievalAdjacencyMatrix,
    ) -> tuple[list[int], list[int], list[int], list[bool]]:
        """Materialize the retrieval branch from the root token.

        Returns ``(tokens, parents, ranks, valid)`` where ``parents`` are
        template-local indices (``0`` = root) and ``valid[i]`` is False when
        the node could not be produced (its parent row was still empty in the
        adjacency matrix).  Nodes whose parent failed are dropped together
        with their whole subtree, so the returned prefix is always
        prefix-closed.
        """
        device = matrix.device
        node_tokens: list[int] = [int(root_token_id)]
        node_valid: list[bool] = [True]
        tokens: list[int] = []
        parents: list[int] = []
        ranks: list[int] = []
        valid: list[bool] = []
        offset = 0
        for width in self.depth_counts:
            width = max(0, min(int(width), self.node_count - offset))
            if width <= 0:
                break
            seg_parents = self.parents[offset : offset + width]
            seg_ranks = self.ranks[offset : offset + width]
            seg_parent_valid = [node_valid[p] for p in seg_parents]
            parent_token_t = torch.tensor(
                [node_tokens[p] for p in seg_parents],
                dtype=torch.long,
                device=device,
            )
            rank_t = torch.tensor(seg_ranks, dtype=torch.long, device=device)
            child_ids = matrix.lookup(parent_token_t, rank_t)
            child_ids_cpu = child_ids.detach().cpu().tolist()
            for j, child_id in enumerate(child_ids_cpu):
                ok = seg_parent_valid[j] and int(child_id) >= 0
                valid.append(ok)
                parents.append(seg_parents[j])
                ranks.append(seg_ranks[j])
                tokens.append(int(child_id) if ok else -1)
                if ok:
                    node_tokens.append(int(child_id))
                    node_valid.append(True)
                else:
                    node_tokens.append(-1)
                    node_valid.append(False)
            offset += width
        return tokens, parents, ranks, valid

    def __repr__(self) -> str:
        return (
            f"RetrievalTemplate(nodes={self.node_count}, "
            f"max_depth={self.max_depth}, k={self.k}, "
            f"depth_counts={self.depth_counts})"
        )


@dataclass
class GraftConfig:
    """Configuration for the prune-then-graft hybrid tree construction.

    ``checkpoints`` are the draft depths at which confidence is evaluated
    (``0`` = the root itself, before any expansion).  When the confidence at a
    checkpoint drops below the corresponding ``thresholds`` entry, the draft
    tree is pruned there, stage ``A`` is entered, only the top
    ``round(stage_draft_fractions[A] * budget)`` draft nodes are kept, and the
    released budget is filled with retrieved nodes.

    The defaults follow the paper's 60-node example (Sec. 3.1): pruning at
    the root / after depth 1 / after depth 5 keeps 8 / 24 / 40 draft nodes
    and assigns 52 / 36 / 20 slots to retrieval (fractions 0.13 / 0.4 / 0.67).
    """

    k: int = 9
    checkpoints: tuple[int, ...] = (0, 1, 5)
    thresholds: tuple[float, ...] = (0.35, 0.25, 0.15)
    stage_draft_fractions: tuple[float, ...] = (0.13, 0.4, 0.67)
    min_template_width: int = 1
    init_from_draft_logits: bool = True

    def __post_init__(self) -> None:
        if len(self.checkpoints) != len(self.thresholds):
            raise ValueError(
                "checkpoints and thresholds must have equal length: "
                f"{self.checkpoints} vs {self.thresholds}"
            )
        if len(self.checkpoints) != len(self.stage_draft_fractions):
            raise ValueError(
                "checkpoints and stage_draft_fractions must have equal length: "
                f"{self.checkpoints} vs {self.stage_draft_fractions}"
            )
        if len({int(c) for c in self.checkpoints}) != len(self.checkpoints):
            raise ValueError(f"checkpoints must be distinct: {self.checkpoints}")
        if any(int(c) < 0 for c in self.checkpoints):
            raise ValueError(f"checkpoints must be non-negative: {self.checkpoints}")
        if any(float(t) <= 0.0 for t in self.thresholds):
            raise ValueError(f"thresholds must be positive: {self.thresholds}")
        if any(not (0.0 < float(f) < 1.0) for f in self.stage_draft_fractions):
            raise ValueError(
                f"stage_draft_fractions must be in (0, 1): {self.stage_draft_fractions}"
            )

    def checkpoint_index(self, depth: int) -> int:
        """Stage index for a checkpoint depth, or -1 if it is not a checkpoint."""
        try:
            return self.checkpoints.index(int(depth))
        except ValueError:
            return -1

    def draft_budget(self, stage: int, budget: int) -> int:
        """Number of draft nodes retained for a pruning stage."""
        fraction = float(self.stage_draft_fractions[int(stage)])
        return max(1, min(int(budget), int(round(fraction * int(budget)))))

    def retrieval_budget(self, stage: int, budget: int) -> int:
        return max(0, int(budget) - self.draft_budget(stage, budget))


def _self_test() -> None:
    """CPU sanity check of the retrieval machinery (no models required)."""
    torch.manual_seed(0)
    vocab, k = 1000, 9
    matrix = RetrievalAdjacencyMatrix(vocab, k, "cpu")
    # Simulate two verification rounds.
    for round_index in range(2):
        n = 8
        parents = torch.randint(0, vocab, (n,))
        logits = torch.randn(n, vocab)
        matrix.update_from_logits(parents, logits)
    assert matrix.matrix.min().item() >= 0, "every updated row must hold valid ids"

    counts = make_decaying_depth_counts(52, 9)
    assert sum(counts) == 52 and counts[0] >= counts[-1], counts
    counts_zero = make_decaying_depth_counts(0, 9)
    assert counts_zero == [0] * 9

    template = RetrievalTemplate(counts, k)
    assert template.node_count == 52, template.node_count
    assert template.max_depth == 9
    assert len(template.node_depths) == 52
    assert max(template.node_depths) == 9
    # prefix-closed parent indices
    for node_index, parent in enumerate(template.parents, start=1):
        assert 0 <= parent < node_index, (node_index, parent)
    assert all(0 <= rank < k for rank in template.ranks)

    tokens, parents, ranks, valid = template.materialize(42, matrix)
    assert len(tokens) == 52
    assert all(0 <= rank < k for rank in ranks)
    # valid nodes form a prefix-closed subtree
    for node_index, (parent, is_valid) in enumerate(zip(parents, valid), start=1):
        if is_valid:
            assert 0 <= parent < node_index
            assert tokens[node_index - 1] >= 0
        else:
            assert tokens[node_index - 1] == -1

    config = GraftConfig()
    assert config.checkpoint_index(1) == 1
    assert config.checkpoint_index(3) == -1
    assert config.draft_budget(0, 60) == 8
    assert config.draft_budget(1, 60) == 24
    assert config.draft_budget(2, 60) == 40
    assert config.retrieval_budget(2, 60) == 20
    print("retrieval self-test OK")


if __name__ == "__main__":
    _self_test()
