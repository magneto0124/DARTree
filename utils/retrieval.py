"""GPU-resident successor table + retrieval-tree topology templates for Graft.

Two decoupled halves make up the retrieval branch (paper Sec. 3.2):

* :class:`GraftAdjacencyMatrix` — the *content*.  A ``[V, k]`` table whose row ``v``
  stores the top-``k`` successors of token ``v`` (token ids).  It is refreshed online
  from the target model's verified-token logits, and one lookup ``M[v, r]`` returns a
  single token.

* :func:`build_retrieval_template` — the *topology*.  A static tree shape deciding
  **which** ``(parent, successor-rank)`` pairs make up the ``K_ret`` retrieved nodes.
  The matrix alone only answers "parent + rank -> token"; it cannot decide the tree
  structure, so a template (a rank-path shape) is required to make the root-centered
  retrieval subtree well-defined.

Both halves are pure ``torch`` eager code and therefore device-agnostic (CUDA / Ascend
NPU), with zero host-device synchronisation.
"""

from __future__ import annotations

import torch


class GraftAdjacencyMatrix:
    """GPU-resident top-k successor table ``M`` of shape ``[V, k]``.

    Row ``v`` stores ``M[v] = argtop_k(p_next | v)`` as token ids.  Uninitialised rows
    are filled with ``pad_token_id`` and flagged ``False`` in :attr:`initialized`; that
    boolean mask (not the pad value) is the source of truth for whether a row is ready,
    because a *legitimately* updated row could in principle contain the pad id.
    """

    def __init__(
        self,
        vocab_size: int,
        k: int,
        device,
        pad_token_id: int = 0,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.k = int(k)
        self.pad_token_id = int(pad_token_id)
        self.device = torch.device(device)
        if self.vocab_size <= 0 or self.k <= 0:
            raise ValueError("vocab_size and k must be positive")
        self.matrix = torch.full(
            (self.vocab_size, self.k),
            self.pad_token_id,
            dtype=dtype,
            device=self.device,
        )
        self.initialized = torch.zeros(
            self.vocab_size, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------ update
    def update(self, token_ids: torch.Tensor, logits: torch.Tensor) -> None:
        """Refresh rows from target-model verified distributions.

        Args:
            token_ids: ``[N]`` int64 — the verified draft/retrieved token of each node.
            logits: ``[N, vocab_size]`` — target next-token logits at those nodes.

        Out-of-range ids (``< 0`` or ``>= vocab_size``) are skipped.  Rows requested by
        multiple nodes are overwritten (last node wins), which is the intended semantics
        since the matrix keeps only the most recent top-k evidence per token.
        """
        token_ids = token_ids.to(self.device, dtype=torch.long).reshape(-1)
        logits = logits.to(self.device).float().reshape(token_ids.numel(), -1)
        if token_ids.numel() == 0:
            return
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"logits vocab dim {logits.shape[-1]} != matrix vocab {self.vocab_size}"
            )

        valid = (token_ids >= 0) & (token_ids < self.vocab_size)
        ids = token_ids[valid]
        if ids.numel() == 0:
            return

        topk = torch.topk(logits[valid], k=self.k, dim=-1).indices
        self.matrix[ids] = topk.to(self.matrix.dtype)
        self.initialized[ids] = True

    # ------------------------------------------------------------------ lookup
    def lookup(self, parent_token_ids: torch.Tensor, ranks) -> torch.Tensor:
        """Return ``M[parent_token_ids, ranks]``.

        Args:
            parent_token_ids: arbitrary-shaped int64 tensor.
            ranks: int / 0-dim tensor / tensor broadcastable against ``parent_token_ids``.
                Must lie in ``[0, k)``.

        Returns:
            Tensor shaped like ``parent_token_ids`` (the rank dim is squashed).

        When ``parent_token_ids`` and ``ranks`` are both ``[N]`` (or broadcastable to a
        common ``[N]``), this is the per-node batched primitive used by BFS retrieval
        expansion — one successor per ``(parent, rank)`` pair.
        """
        parent_token_ids = parent_token_ids.to(self.device, dtype=torch.long)
        if torch.is_tensor(ranks):
            rank_idx = ranks.to(self.device, dtype=torch.long)
        else:
            rank_idx = torch.as_tensor(ranks, dtype=torch.long, device=self.device)
        if torch.is_tensor(ranks):
            lo = int(rank_idx.min().item()) if rank_idx.numel() > 0 else 0
            hi = int(rank_idx.max().item()) if rank_idx.numel() > 0 else 0
            if lo < 0 or hi >= self.k:
                raise IndexError(
                    f"rank out of range [{lo}, {hi}]: matrix has k={self.k} successors per row"
                )
        else:
            r = int(rank_idx.item())
            if r < 0 or r >= self.k:
                raise IndexError(f"rank {r} out of range: matrix has k={self.k} successors")
        rows = self.matrix[parent_token_ids]  # [..., k]
        return rows.gather(-1, rank_idx.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------ readiness
    def is_ready(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask: whether each requested row has been initialised."""
        token_ids = token_ids.to(self.device, dtype=torch.long)
        return self.initialized[token_ids]

    def ready_rows(self) -> int:
        """Number of initialised rows (for logging/stats only)."""
        return int(self.initialized.sum().item())

    # ------------------------------------------------------------- (de)serialise
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"matrix": self.matrix, "initialized": self.initialized}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.matrix.copy_(state["matrix"])
        self.initialized.copy_(state["initialized"])


def build_retrieval_template(
    level_widths,
) -> tuple[list[int], list[int], list[int]]:
    """Static rank-path template for the root-centered retrieval subtree.

    Args:
        level_widths: ``[w1, w2, ..., wD]`` node counts per depth (depth is 1-based).

    Returns three parallel lists in BFS order::

        parents[i]  -> template index of the parent (``-1`` = the shared root token)
        ranks[i]    -> successor rank; the node's token is ``M[parent_token, ranks[i]]``
        depths[i]   -> 1-based depth

    Construction rules:

    * Depth-1 nodes are children of the shared root with ranks ``0..w1-1``.
    * Each deeper layer is assigned round-robin over the previous layer **in rank
      order**, so the top-ranked parent receives its next child before lower-ranked
      parents.  Because the rank-0 node is always first in every frontier, the
      ``rank0 -> rank0 -> ...`` greedy chain automatically reaches the full depth —
      matching the "top-ranked successors extend deeper" shape in paper Appendix A.1.

    The result is always a valid prefix-closed tree: every non-root parent index is
    strictly smaller than the child index, so it converts to a legal DARTree
    ``parents`` list (root sentinel aside) without further repair.
    """
    parents: list[int] = []
    ranks: list[int] = []
    depths: list[int] = []
    frontier: list[int] = []  # template indices of the previous layer, in rank order

    for depth, width in enumerate(level_widths, start=1):
        width = int(width)
        if width <= 0:
            break
        layer_start = len(parents)
        if depth == 1:
            parent_list = [-1] * width
        else:
            if not frontier:
                break  # cannot grow deeper without any parents
            parent_list = [frontier[i % len(frontier)] for i in range(width)]

        child_counter: dict[int, int] = {}
        for parent in parent_list:
            rank = child_counter.get(parent, 0)
            child_counter[parent] = rank + 1
            parents.append(parent)
            ranks.append(rank)
            depths.append(depth)
        frontier = list(range(layer_start, layer_start + width))

    return parents, ranks, depths


def default_level_widths(
    budget: int, max_depth: int, root_width: int = 8
) -> list[int]:
    """Turn a retrieval node budget into a front-loaded per-depth width schedule.

    Level 1 gets ``root_width`` (capped so every remaining level keeps >=1 node), and
    the rest of the budget is then split as evenly as possible over the remaining
    levels.  The total always equals ``budget`` (when ``0 < budget``) and the length is
    at most ``max_depth``.  Callers wanting the paper's exact shapes (Appendix A.1 /
    Table 7, e.g. ``[8, 10, 8, 6, 5, 4, 4, 4, 3]``) should pass an explicit
    ``level_widths`` list to :func:`build_retrieval_template` instead.
    """
    budget = int(budget)
    max_depth = int(max_depth)
    if budget <= 0 or max_depth <= 0:
        return []
    depth = min(max_depth, budget)  # each level needs at least one node
    if depth == 1:
        return [budget]

    w1 = min(max(1, int(root_width)), budget - (depth - 1))
    rest = budget - w1
    base, extra = divmod(rest, depth - 1)
    widths = [w1] + [base + (1 if i < extra else 0) for i in range(depth - 1)]
    return widths