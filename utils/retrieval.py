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

from typing import Any

import torch


def resolve_graft_retain(
    budget: int, ratio: float, supertree_node_count: int | None = None
) -> tuple[int, int]:
    """Split a fixed verification budget between draft and retrieval subtrees.

    This is the V1 fixed-ratio budget bookkeeping: keep ``round(ratio * budget)``
    draft nodes after Top-B pruning and give the rest to the retrieval subtree, so the
    total never exceeds ``budget`` (``draft_retain + k_ret == budget``).

    Args:
        budget: total verification budget ``K_max`` (e.g. ``tree_budget``).
        ratio: fraction of ``budget`` retained for the draft tree, in ``(0, 1]``.
        supertree_node_count: number of nodes the supertree actually grew.  When given,
            ``draft_retain`` is clamped so it never exceeds the available supertree
            nodes (the frontier may have stopped early).

    Returns:
        ``(draft_retain, k_ret)`` — the Top-B pruning target and the retrieved-node
        budget respectively.
    """
    budget = int(budget)
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    draft_retain = max(1, round(ratio * budget))
    if supertree_node_count is not None:
        # The supertree may have fewer nodes than requested; never claim more than
        # exist.  Keep the result >= 1 so the tree always has at least the root path.
        draft_retain = max(1, min(int(draft_retain), int(supertree_node_count)))
    k_ret = budget - draft_retain
    return draft_retain, k_ret


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

        Tie-breaking: when several vocab positions share the top-k boundary value,
        ``torch.topk`` picks among them **arbitrarily** (implementation-defined, and
        not necessarily the smallest index), so a row's contents are only determined
        up to the tie set.  This is harmless for real decoding (continuous float
        logits practically never tie exactly) and matches the ``argtop_k`` contract,
        which is itself multi-valued under ties.  Tests must therefore construct
        logits with strictly ranked values when asserting exact rows.
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
        # Scalar-rank broadcast: a single rank (int or 0-dim tensor) applies to
        # every parent.  broadcast_to also guarantees the gather index has the
        # same number of dims as `rows` (parent [N] + scalar rank -> [N, 1]
        # index against [N, k] input); an un-broadcastable ranks tensor raises
        # instead of silently truncating.
        rank_idx = torch.broadcast_to(rank_idx, parent_token_ids.shape)
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

    The root-centered retrieval subtree lives or dies by breadth at level 1, so
    the root layer is allocated **first**: ``w1`` gets up to ``root_width``
    nodes (capped at ``budget - 1`` so the deeper levels keep at least one
    node), and only the remainder is spread over the deeper levels.  This keeps
    a small budget from degenerating into a single rank-0 chain when
    ``max_depth >= budget`` (previously ``depth = min(max_depth, budget)``
    forced ``w1 = 1`` in that case).  The total always equals ``budget`` (when
    ``0 < budget``) and the length is at most ``max_depth``.  Callers wanting
    the paper's exact shapes (Appendix A.1 / Table 7, e.g.
    ``[8, 10, 8, 6, 5, 4, 4, 4, 3]``) should pass an explicit ``level_widths``
    list to :func:`build_retrieval_template` instead.
    """
    budget = int(budget)
    max_depth = int(max_depth)
    if budget <= 0 or max_depth <= 0:
        return []
    if budget == 1 or max_depth == 1:
        return [budget]
    w1 = min(max(1, int(root_width)), budget - 1)
    rest = budget - w1
    depth = min(max_depth - 1, rest)  # deeper levels (each keeps >= 1 node)
    base, extra = divmod(rest, depth)
    widths = [w1] + [base + (1 if i < extra else 0) for i in range(depth)]
    return widths


def build_retrieval_subtree(
    root_token_id: int,
    matrix: GraftAdjacencyMatrix,
    template: tuple[list[int], list[int], list[int]],
    k_ret: int,
) -> dict[str, Any]:
    """Build the root-centered retrieval subtree ``G_ret^s`` from the matrix.

    Walks the rank-path ``template`` in BFS order and materialises node tokens
    with ``token = M[parent_token, rank]``.  Node 0 of the result is the shared
    root (token ``root_token_id``), mirroring the draft tree's layout, so the
    subtree can be merged by :func:`graft_hybrid_tree` under one shared root.

    Fallbacks (keep the tree legal and prefix-closed):

    * A node whose parent row yields no usable successor (uninitialised row, or
      every rank resolving to the pad sentinel / an out-of-range id) is dropped
      together with its whole descendant subtree — a cold ``M`` simply produces
      fewer retrieved nodes.
    * Successor ranks are scanned upward from the template rank, skipping
      invalid sentinels and tokens already used by another child of the same
      parent **within this subtree**, so the result is always a valid tree.
    * The node budget is capped at ``k_ret`` (BFS order == template priority
      order, since :func:`build_retrieval_template` front-loads the rank-0
      chain).

    Args:
        root_token_id: the current-round root token ``x_t`` (shared with the
            draft tree; its row in ``M`` is the root-level lookup table).
        matrix: the GPU-resident adjacency matrix.
        template: ``(parents, ranks, depths)`` from
            :func:`build_retrieval_template`.
        k_ret: maximum number of retrieved nodes to produce.

    Returns:
        A dict with ``token_ids`` (nodes 1..m), ``depths`` (1-based), ``parents``
        (local indices, ``parents[0] == -1``), ``ranks`` (successor rank actually
        used per node) and ``stats``.  Pure torch eager code, CPU/GPU alike.
    """
    tpl_parents, tpl_ranks, tpl_depths = template
    k_ret = max(0, int(k_ret))
    pad_token = int(matrix.pad_token_id)
    vocab_size = int(matrix.vocab_size)
    k = int(matrix.k)

    stats = {
        "template_node_count": float(len(tpl_parents)),
        "retrieved_node_count": 0.0,
        "dropped_node_count": 0.0,
        "rank_shifted_node_count": 0.0,
        "retrieval_hit_rate": 0.0,
    }
    if k_ret == 0 or not tpl_parents:
        return {
            "token_ids": [],
            "depths": [],
            "parents": [-1],
            "ranks": [],
            "stats": stats,
        }

    token_ids: list[int] = []
    depths: list[int] = []
    ranks: list[int] = []
    parents: list[int] = [-1]  # local index 0 = shared root
    node_tokens: list[int] = [int(root_token_id)]
    child_token_sets: list[set[int]] = [set()]
    tpl_to_local: dict[int, int] = {}

    for index, (tpl_parent, tpl_rank, depth) in enumerate(
        zip(tpl_parents, tpl_ranks, tpl_depths)
    ):
        if len(token_ids) >= k_ret:
            break
        if tpl_parent == -1:
            local_parent = 0
        else:
            local_parent = tpl_to_local.get(tpl_parent)
            if local_parent is None:
                # Parent was dropped -> its whole descendant subtree is dropped.
                stats["dropped_node_count"] += 1.0
                continue
        parent_token = node_tokens[local_parent]
        used_rank = None
        for rank in range(int(tpl_rank), k):
            token = int(
                matrix.lookup(
                    torch.tensor([parent_token], device=matrix.device), rank
                ).item()
            )
            if token < 0 or token >= vocab_size or token == pad_token:
                continue
            if token in child_token_sets[local_parent]:
                continue
            used_rank = rank
            break
        if used_rank is None:
            stats["dropped_node_count"] += 1.0
            continue
        if used_rank != int(tpl_rank):
            stats["rank_shifted_node_count"] += 1.0
        local_index = len(node_tokens)
        tpl_to_local[index] = local_index
        node_tokens.append(token)
        parents.append(local_parent)
        token_ids.append(token)
        depths.append(int(depth))
        ranks.append(used_rank)
        child_token_sets[local_parent].add(token)
        child_token_sets.append(set())

    stats["retrieved_node_count"] = float(len(token_ids))
    attempted = min(len(tpl_parents), k_ret)
    stats["retrieval_hit_rate"] = (
        float(len(token_ids)) / float(attempted) if attempted > 0 else 0.0
    )
    return {
        "token_ids": token_ids,
        "depths": depths,
        "parents": parents,
        "ranks": ranks,
        "stats": stats,
    }


def _build_visibility(parents: list[int]) -> torch.Tensor:
    """Pure-torch rebuild of the ancestor-visibility matrix from ``parents``.

    Mirrors ``eval_dartree.build_visibility`` (same recurrence) without pulling
    a numpy dependency into this module; the result is a CPU bool tensor shaped
    ``[len(parents), len(parents)]``.
    """
    size = len(parents)
    visibility = torch.zeros((size, size), dtype=torch.bool)
    if size == 0:
        return visibility
    visibility[0, 0] = True
    for index in range(1, size):
        parent_index = int(parents[index])
        visibility[index, :index] = visibility[parent_index, :index]
        visibility[index, index] = True
    return visibility


def graft_hybrid_tree(
    draft_tree: dict[str, Any],
    retrieval_tree: dict[str, Any],
    k_max: int,
    matrix: GraftAdjacencyMatrix | None = None,
    root_token_id: int | None = None,
    dedup: str = "skip",
    slots: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Merge the draft tree with the retrieval subtree under one shared root.

    Node 0 (root) is shared: retrieval depth-1 nodes become extra siblings of the
    root next to the retained draft children, and deeper retrieval nodes hang
    under them (the plan's "root 中心检索子树").  Draft node indices are
    preserved; retrieval nodes are appended afterwards.

    Deduplication — a retrieval token already present as a child of the same
    merged parent would overwrite ``child_maps``' single-value index and break
    ``follow_verified_tree``:

    * ``dedup="skip"`` (default, plan §4.3 方案 A): rescue the node by
      re-scanning successor ranks with the matrix (requires ``matrix`` and
      ``root_token_id``); when no usable rank remains, drop the node and its
      whole descendant subtree.
    * ``dedup="redirect"`` (方案 B): map the node onto the existing child — its
      descendants attach under that child, preserving the retrieval budget.

    ``slots`` selects the *tree-internal* ablation (plan §4.2 "逐空位填充 /
    树内嫁接", a non-paper Graft(TAIL)-like variant, for comparison only):
    when given as ``[(parent, depth), ...]`` (released pruned slots in the
    draft tree's index space), retrieval tokens are looked up per slot parent
    (``token = M[parent_token, rank]``) and attached **under that parent at the
    pruned node's depth**, instead of building a root-centered retrieval
    subtree.  ``k_max`` still bounds the merged node count.

    Args:
        draft_tree: output-shaped dict with ``node_token_ids`` (list),
            ``node_depths`` (list) and ``parents`` (list, ``parents[0] == -1``);
            ``child_maps``/``visibility`` are optional (rebuilt).
        retrieval_tree: output of :func:`build_retrieval_subtree` (unused when
            ``slots`` is given; its ``stats`` are still carried over).
        k_max: total verification budget ``K_max``; the merged node count must
            not exceed it (defensive check — ``draft_retain + k_ret <= K_max``
            holds by construction).
        matrix: adjacency matrix, needed for the 方案 A rank re-scan and for the
            ``slots`` lookups.
        root_token_id: token of the shared root, needed for root-level lookups.
        dedup: ``"skip"`` (default) or ``"redirect"``.
        slots: optional ``[(parent, depth), ...]`` pruned-slot list for the
            ``into_slot`` ablation; ``None`` selects the root-centered merge.

    Returns:
        Merged tree with ``node_token_ids``/``node_depths``/``parents``/
        ``child_maps``/``visibility`` plus ``stats``.
    """
    draft_tokens = [int(x) for x in draft_tree["node_token_ids"]]
    draft_depths = [int(x) for x in draft_tree["node_depths"]]
    draft_parents = [int(x) for x in draft_tree["parents"]]
    ret_tokens = [int(x) for x in retrieval_tree["token_ids"]]
    ret_depths = [int(x) for x in retrieval_tree["depths"]]
    ret_parents = [int(x) for x in retrieval_tree["parents"]]
    ret_ranks = [int(x) for x in retrieval_tree.get("ranks", [])]
    dedup = dedup if dedup in ("skip", "redirect") else "skip"

    stats: dict[str, float] = dict(retrieval_tree.get("stats", {}))
    stats.setdefault("retrieved_node_count", 0.0)
    stats["draft_node_count"] = float(len(draft_tokens))
    stats["dedup_skipped_nodes"] = 0.0
    stats["dedup_redirected_nodes"] = 0.0
    stats["graft_rank_rescanned_nodes"] = 0.0

    if slots is not None:
        # into_slot ablation: fill the released pruned slots with retrieval
        # tokens under the pruned nodes' original (kept) parents.
        return _merge_into_slots(
            draft_parents,
            draft_tokens,
            draft_depths,
            slots,
            k_max,
            matrix,
            root_token_id,
            dedup,
            stats,
        )

    # Identity: no retrieval budget -> the draft tree is returned untouched
    # (equivalent to the `pruned` variant, lossless).
    if not ret_tokens:
        return {
            "node_token_ids": draft_tokens,
            "node_depths": draft_depths,
            "parents": draft_parents,
            "child_maps": _rebuild_child_maps(draft_parents, draft_tokens),
            "visibility": _build_visibility(draft_parents),
            "stats": stats,
        }

    merged_parents = list(draft_parents)
    merged_tokens = list(draft_tokens)
    merged_depths = list(draft_depths)
    merged_child_maps = _rebuild_child_maps(draft_parents, draft_tokens)

    pad_token = int(matrix.pad_token_id) if matrix is not None else None
    vocab_size = int(matrix.vocab_size) if matrix is not None else None
    k = int(matrix.k) if matrix is not None else None

    ret_to_merged: dict[int, int] = {}
    for ret_index in range(1, len(ret_parents)):
        ret_parent = int(ret_parents[ret_index])
        if ret_parent == 0:
            merged_parent = 0
        else:
            merged_parent = ret_to_merged.get(ret_parent)
            if merged_parent is None:
                # Descendant of a dropped node (should not happen with a
                # prefix-closed retrieval tree; defensive).
                stats["dedup_skipped_nodes"] += 1.0
                continue
        token = int(ret_tokens[ret_index - 1])
        existing = merged_child_maps[merged_parent].get(token)
        if existing is not None:
            if dedup == "redirect":
                # 方案 B: attach this node's descendants under the existing
                # child; the node itself adds no new merged index.
                ret_to_merged[ret_index] = existing
                stats["dedup_redirected_nodes"] += 1.0
                continue
            # 方案 A: try to rescue the node by re-scanning successor ranks.
            if matrix is not None and root_token_id is not None:
                if merged_parent == 0:
                    parent_token = int(root_token_id)
                else:
                    parent_token = int(merged_tokens[merged_parent - 1])
                base_rank = (
                    ret_ranks[ret_index - 1]
                    if ret_index - 1 < len(ret_ranks)
                    else 0
                )
                rescued = None
                for rank in range(base_rank + 1, k):
                    candidate = int(
                        matrix.lookup(
                            torch.tensor([parent_token], device=matrix.device),
                            rank,
                        ).item()
                    )
                    if (
                        candidate < 0
                        or candidate >= vocab_size
                        or candidate == pad_token
                        or candidate in merged_child_maps[merged_parent]
                    ):
                        continue
                    rescued = candidate
                    break
                if rescued is None:
                    stats["dedup_skipped_nodes"] += 1.0
                    continue
                token = rescued
                stats["graft_rank_rescanned_nodes"] += 1.0
            else:
                stats["dedup_skipped_nodes"] += 1.0
                continue

        new_index = len(merged_parents)
        merged_parents.append(merged_parent)
        merged_tokens.append(token)
        merged_depths.append(int(ret_depths[ret_index - 1]))
        merged_child_maps.append(dict())
        merged_child_maps[merged_parent][token] = new_index
        ret_to_merged[ret_index] = new_index

    merged_count = len(merged_tokens)
    if merged_count > int(k_max):
        raise RuntimeError(
            f"grafted tree has {merged_count} nodes, exceeding k_max={k_max}"
        )

    stats["graft_merged_retrieval_nodes"] = float(
        merged_count - len(draft_tokens)
    )
    stats["graft_total_nodes"] = float(merged_count)
    stats["node_count"] = float(merged_count)
    stats["tree_height"] = float(max(merged_depths) if merged_depths else 0.0)
    return {
        "node_token_ids": merged_tokens,
        "node_depths": merged_depths,
        "parents": merged_parents,
        "child_maps": merged_child_maps,
        "visibility": _build_visibility(merged_parents),
        "stats": stats,
    }


def _rebuild_child_maps(
    parents: list[int], token_ids: list[int]
) -> list[dict[int, int]]:
    """Rebuild ``child_maps`` (list of per-parent token->child dicts)."""
    child_maps: list[dict[int, int]] = [
        dict() for _ in range(len(parents))
    ]
    for node_index, token_id in enumerate(token_ids, start=1):
        parent_index = int(parents[node_index])
        child_maps[parent_index][int(token_id)] = int(node_index)
    return child_maps


def _merge_into_slots(
    draft_parents: list[int],
    draft_tokens: list[int],
    draft_depths: list[int],
    slots: list[tuple[int, int]],
    k_max: int,
    matrix: GraftAdjacencyMatrix | None,
    root_token_id: int | None,
    dedup: str,
    stats: dict[str, float],
) -> dict[str, Any]:
    """Graft ablation: fill pruned slots with retrieval tokens (plan §4.2).

    Each ``(parent, depth)`` slot is a pruned node's released position: the slot
    parent was kept by Top-B pruning, the slot node itself was not.  A retrieval
    token ``M[parent_token, rank]`` is attached under that parent at the same
    depth (per-parent successor ranks 0, 1, 2, ...; rank is advanced past
    invalid sentinels and sibling duplicates).  Slot nodes are leaves — the
    pruned nodes' descendants were pruned too, so no subtree bookkeeping is
    needed here; ``dedup`` is accepted for signature symmetry but the rank scan
    already prevents collisions (always "skip"-like behaviour).
    """
    merged_parents = list(draft_parents)
    merged_tokens = list(draft_tokens)
    merged_depths = list(draft_depths)
    merged_child_maps = _rebuild_child_maps(draft_parents, draft_tokens)

    pad_token = int(matrix.pad_token_id) if matrix is not None else None
    vocab_size = int(matrix.vocab_size) if matrix is not None else None
    k = int(matrix.k) if matrix is not None else None

    rank_counters: dict[int, int] = {}
    filled = 0
    for slot_parent, slot_depth in slots:
        slot_parent = int(slot_parent)
        slot_depth = int(slot_depth)
        if slot_parent == 0:
            expected_depth = 1
        else:
            expected_depth = int(merged_depths[slot_parent - 1]) + 1
        if slot_depth != expected_depth:
            raise ValueError(
                f"slot depth {slot_depth} inconsistent with parent {slot_parent} "
                f"(expected {expected_depth})"
            )
        if matrix is None or root_token_id is None:
            # No matrix -> cannot look up a slot token; count as skipped.
            stats["dedup_skipped_nodes"] += 1.0
            continue
        if slot_parent == 0:
            parent_token = int(root_token_id)
        else:
            parent_token = int(merged_tokens[slot_parent - 1])
        rank = rank_counters.get(slot_parent, 0)
        placed_token = None
        while rank < k:
            candidate = int(
                matrix.lookup(
                    torch.tensor([parent_token], device=matrix.device), rank
                ).item()
            )
            if (
                candidate < 0
                or candidate >= vocab_size
                or candidate == pad_token
                or candidate in merged_child_maps[slot_parent]
            ):
                rank += 1
                continue
            placed_token = candidate
            break
        if placed_token is None:
            stats["dedup_skipped_nodes"] += 1.0
            continue
        rank_counters[slot_parent] = rank + 1
        new_index = len(merged_parents)
        merged_parents.append(slot_parent)
        merged_tokens.append(placed_token)
        merged_depths.append(slot_depth)
        merged_child_maps.append(dict())
        merged_child_maps[slot_parent][placed_token] = new_index
        filled += 1

    merged_count = len(merged_tokens)
    if merged_count > int(k_max):
        raise RuntimeError(
            f"grafted tree has {merged_count} nodes, exceeding k_max={k_max}"
        )

    stats["graft_slot_filled_nodes"] = float(filled)
    stats["graft_merged_retrieval_nodes"] = float(filled)
    stats["graft_total_nodes"] = float(merged_count)
    stats["node_count"] = float(merged_count)
    stats["tree_height"] = float(max(merged_depths) if merged_depths else 0.0)
    return {
        "node_token_ids": merged_tokens,
        "node_depths": merged_depths,
        "parents": merged_parents,
        "child_maps": merged_child_maps,
        "visibility": _build_visibility(merged_parents),
        "stats": stats,
    }