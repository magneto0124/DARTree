"""Tests for Graft Phase 2: retrieval subtree + shared-root hybrid merge.

Pure-torch, CPU-runnable; no model weights required.  `eval_dartree` cannot be
imported here (it pulls transformers/triton), so the reference visibility
recurrence is re-implemented locally — it is the same 4-line recurrence used by
``eval_dartree.build_visibility``.
"""
from __future__ import annotations

import pytest
import torch

from utils.retrieval import (
    GraftAdjacencyMatrix,
    build_retrieval_subtree,
    build_retrieval_template,
    graft_hybrid_tree,
)


def reference_visibility(parents: list[int]) -> torch.Tensor:
    """Copy of the ``eval_dartree.build_visibility`` recurrence."""
    size = len(parents)
    visibility = torch.zeros((size, size), dtype=torch.bool)
    visibility[0, 0] = True
    for index in range(1, size):
        parent_index = int(parents[index])
        visibility[index, :index] = visibility[parent_index, :index]
        visibility[index, index] = True
    return visibility


def assert_valid_tree(result: dict) -> None:
    """Common invariants: prefix-closure, child_maps consistency, visibility."""
    parents = [int(x) for x in result["parents"]]
    tokens = [int(x) for x in result["node_token_ids"]]
    depths = [int(x) for x in result["node_depths"]]
    assert len(parents) == len(tokens) + 1 == len(depths) + 1
    assert parents[0] == -1
    for index in range(1, len(parents)):
        assert 0 <= parents[index] < index, "parents must be prefix-closed"
    # child_maps rebuilt from parents+tokens must match exactly.
    rebuilt = [dict() for _ in range(len(parents))]
    for index, token in enumerate(tokens, start=1):
        rebuilt[parents[index]][token] = index
    assert rebuilt == result["child_maps"]
    # Visibility must match the reference recurrence on the merged parents.
    assert torch.equal(
        result["visibility"], reference_visibility(parents)
    )
    # Depths must be consistent with parent depths.
    for index in range(1, len(parents)):
        if parents[index] == 0:
            assert depths[index - 1] == 1
        else:
            assert depths[index - 1] == depths[parents[index] - 1] + 1


def make_matrix(
    vocab_size: int = 16, k: int = 3, rows: dict[int, list[int]] | None = None
) -> GraftAdjacencyMatrix:
    m = GraftAdjacencyMatrix(
        vocab_size=vocab_size, k=k, device="cpu", pad_token_id=0
    )
    if rows:
        ids = torch.tensor(list(rows.keys()))
        logits = torch.zeros(len(ids), vocab_size)
        for row, (vocab_id, successors) in enumerate(rows.items()):
            logits[row, successors] = torch.linspace(
                len(successors), 1.0, len(successors)
            )
        m.update(ids, logits)
    return m


def test_build_retrieval_subtree_basic():
    # Root 0 -> [7, 2, 9]; token 7 -> [3, 4, 5]; token 2 -> [8, 1, 6].
    m = make_matrix(rows={0: [7, 2, 9], 7: [3, 4, 5], 2: [8, 1, 6]})
    parents, ranks, depths = build_retrieval_template([2, 2])
    tree = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=10)
    assert tree["token_ids"] == [7, 2, 3, 8]
    assert tree["depths"] == [1, 1, 2, 2]
    assert tree["parents"] == [-1, 0, 0, 1, 2]
    assert tree["ranks"] == [0, 1, 0, 0]
    assert tree["stats"]["retrieved_node_count"] == 4.0
    assert tree["stats"]["retrieval_hit_rate"] == 1.0
    assert tree["stats"]["dropped_node_count"] == 0.0
    # Prefix-closed: parent local index < child local index.
    for index in range(1, len(tree["parents"])):
        assert 0 <= tree["parents"][index] < index


def test_build_retrieval_subtree_empty_matrix_degrades_gracefully():
    m = make_matrix()  # no rows initialised -> every lookup is the pad sentinel
    parents, ranks, depths = build_retrieval_template([3, 2])
    tree = build_retrieval_subtree(5, m, (parents, ranks, depths), k_ret=5)
    assert tree["token_ids"] == []
    assert tree["parents"] == [-1]
    assert tree["stats"]["retrieved_node_count"] == 0.0
    assert tree["stats"]["retrieval_hit_rate"] == 0.0
    assert tree["stats"]["dropped_node_count"] == 5.0


def test_build_retrieval_subtree_budget_cap():
    m = make_matrix(rows={0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})
    parents, ranks, depths = build_retrieval_template([2, 2])
    tree = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    assert tree["stats"]["retrieved_node_count"] == 2.0
    assert tree["token_ids"] == [1, 2]
    assert tree["parents"] == [-1, 0, 0]


def test_build_retrieval_subtree_pad_fallback_advances_rank():
    # Root row is ready but its rank-0 successor is the pad sentinel (legit
    # pad in top-k); the builder must advance to rank 1.
    m = make_matrix(vocab_size=8, k=2, rows={0: [0, 3]})
    parents, ranks, depths = build_retrieval_template([2])
    tree = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    assert tree["token_ids"] == [3]
    assert tree["ranks"] == [1]
    assert tree["stats"]["rank_shifted_node_count"] == 1.0
    # The other depth-1 slot could not find a valid successor (rank 2 >= k).
    assert tree["stats"]["dropped_node_count"] == 1.0


def test_build_retrieval_subtree_dropped_parent_kills_subtree():
    # Only the root row is ready; depth-2 parents (tokens 1, 2) are unready, so
    # their children must be dropped with them.
    m = make_matrix(rows={0: [1, 2]})
    parents, ranks, depths = build_retrieval_template([2, 2])
    tree = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=10)
    assert tree["token_ids"] == [1, 2]
    assert tree["depths"] == [1, 1]
    assert tree["stats"]["retrieved_node_count"] == 2.0
    assert tree["stats"]["dropped_node_count"] == 2.0


def test_graft_hybrid_tree_shared_root_merge():
    draft = {
        "node_token_ids": [1, 5, 7],
        "node_depths": [1, 1, 2],
        "parents": [-1, 0, 0, 1],
    }
    m = make_matrix(rows={0: [2, 3, 4], 2: [6, 8, 9]})
    parents, ranks, depths = build_retrieval_template([2, 1])
    ret = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    merged = graft_hybrid_tree(draft, ret, k_max=10)
    assert_valid_tree(merged)
    # Draft nodes keep their indices; retrieval nodes are appended after.
    assert merged["node_token_ids"][:3] == [1, 5, 7]
    assert merged["parents"][:4] == [-1, 0, 0, 1]
    assert merged["node_token_ids"][3:] == [2, 3]
    assert merged["node_depths"][3:] == [1, 1]
    assert merged["stats"]["graft_merged_retrieval_nodes"] == 2.0
    assert merged["stats"]["graft_total_nodes"] == 5.0
    assert merged["stats"]["dedup_skipped_nodes"] == 0.0


def test_graft_hybrid_tree_identity_when_no_retrieval():
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    empty = {
        "token_ids": [],
        "depths": [],
        "parents": [-1],
        "ranks": [],
        "stats": {"retrieved_node_count": 0.0},
    }
    merged = graft_hybrid_tree(draft, empty, k_max=64)
    assert merged["node_token_ids"] == [1, 5]
    assert merged["parents"] == [-1, 0, 0]
    assert_valid_tree(merged)


def test_graft_hybrid_tree_dedup_skip_drops_without_matrix():
    # Retrieval depth-1 token 1 collides with an existing draft child of the
    # root; without a matrix there is no way to re-scan ranks -> node dropped.
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    m = make_matrix(rows={0: [1, 2, 3]})
    parents, ranks, depths = build_retrieval_template([2])
    ret = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    merged = graft_hybrid_tree(draft, ret, k_max=10)
    assert_valid_tree(merged)
    assert merged["node_token_ids"] == [1, 5, 2]  # 1 dropped, 2 kept
    assert merged["stats"]["dedup_skipped_nodes"] == 1.0


def test_graft_hybrid_tree_dedup_skip_rescues_with_matrix():
    # Same collision as above, but now the matrix is available for 方案 A rank
    # re-scanning: rank 1 yields 2, which is not taken -> node survives.
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    m = make_matrix(rows={0: [1, 2, 3]})
    parents, ranks, depths = build_retrieval_template([2])
    ret = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    merged = graft_hybrid_tree(
        draft, ret, k_max=10, matrix=m, root_token_id=0, dedup="skip"
    )
    assert_valid_tree(merged)
    # Node 1 rescues to token 2; node 2 then collides with that rescue and
    # rescues to token 3.
    assert merged["node_token_ids"] == [1, 5, 2, 3]
    assert merged["stats"]["dedup_skipped_nodes"] == 0.0
    assert merged["stats"]["graft_rank_rescanned_nodes"] == 2.0


def test_graft_hybrid_tree_dedup_redirect():
    # 方案 B: the colliding node maps onto the existing child; its descendants
    # attach under that child instead of being dropped.
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    m = make_matrix(rows={0: [1, 2, 3], 1: [4, 6, 7]})
    parents, ranks, depths = build_retrieval_template([2, 1])
    ret = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=3)
    merged = graft_hybrid_tree(
        draft, ret, k_max=10, dedup="redirect"
    )
    assert_valid_tree(merged)
    # node 1 collides -> redirected to existing draft child 1; its child
    # (token 4, parent = template node 1) attaches under merged node 1.
    assert merged["node_token_ids"] == [1, 5, 2, 4]
    assert merged["parents"] == [-1, 0, 0, 0, 1]
    assert merged["node_depths"] == [1, 1, 1, 2]
    assert merged["stats"]["dedup_redirected_nodes"] == 1.0
    assert merged["stats"]["dedup_skipped_nodes"] == 0.0


def test_graft_hybrid_tree_kmax_guard_raises():
    draft = {
        "node_token_ids": [1, 5, 7, 8, 9],
        "node_depths": [1, 1, 2, 2, 2],
        "parents": [-1, 0, 0, 1, 1, 2],
    }
    m = make_matrix(rows={0: [2, 3]})
    parents, ranks, depths = build_retrieval_template([2])
    ret = build_retrieval_subtree(0, m, (parents, ranks, depths), k_ret=2)
    with pytest.raises(RuntimeError):
        graft_hybrid_tree(draft, ret, k_max=6)


# ----------------------------------------------------------------------
# into_slot ablation (plan §4.2 "逐空位填充 / 树内嫁接")


def empty_retrieval_tree() -> dict:
    return {
        "token_ids": [],
        "depths": [],
        "parents": [-1],
        "ranks": [],
        "stats": {},
    }


def test_graft_hybrid_tree_into_slot_fills_pruned_slots():
    # Draft: root children 1, 5; node 3 is a child of node 1.  Slots released
    # by pruning: (root, depth 1) and (node 1, depth 2).
    draft = {
        "node_token_ids": [1, 5, 7],
        "node_depths": [1, 1, 2],
        "parents": [-1, 0, 0, 1],
    }
    m = make_matrix(rows={0: [2, 3, 4], 1: [6, 8, 9]})
    merged = graft_hybrid_tree(
        draft,
        empty_retrieval_tree(),
        k_max=10,
        matrix=m,
        root_token_id=0,
        slots=[(0, 1), (1, 2)],
    )
    assert_valid_tree(merged)
    assert merged["node_token_ids"] == [1, 5, 7, 2, 6]
    assert merged["parents"] == [-1, 0, 0, 1, 0, 1]
    assert merged["node_depths"] == [1, 1, 2, 1, 2]
    assert merged["stats"]["graft_slot_filled_nodes"] == 2.0
    assert merged["stats"]["dedup_skipped_nodes"] == 0.0


def test_graft_hybrid_tree_into_slot_rank_advance_and_drop():
    # Slot 1's rank-0 token 1 collides with an existing draft child -> advances
    # to rank 1 (token 9).  Slot 2 then has no rank left (k=2) -> dropped.
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    m = make_matrix(vocab_size=10, k=2, rows={0: [1, 9]})
    merged = graft_hybrid_tree(
        draft,
        empty_retrieval_tree(),
        k_max=10,
        matrix=m,
        root_token_id=0,
        slots=[(0, 1), (0, 1)],
    )
    assert_valid_tree(merged)
    assert merged["node_token_ids"] == [1, 5, 9]
    assert merged["node_depths"] == [1, 1, 1]
    assert merged["stats"]["graft_slot_filled_nodes"] == 1.0
    assert merged["stats"]["dedup_skipped_nodes"] == 1.0


def test_graft_hybrid_tree_into_slot_without_matrix_skips_all():
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    merged = graft_hybrid_tree(
        draft,
        empty_retrieval_tree(),
        k_max=10,
        slots=[(0, 1), (1, 2)],
    )
    assert_valid_tree(merged)
    assert merged["node_token_ids"] == [1, 5]
    assert merged["stats"]["graft_slot_filled_nodes"] == 0.0
    assert merged["stats"]["dedup_skipped_nodes"] == 2.0


def test_graft_hybrid_tree_into_slot_depth_mismatch_raises():
    draft = {
        "node_token_ids": [1, 5],
        "node_depths": [1, 1],
        "parents": [-1, 0, 0],
    }
    m = make_matrix(rows={0: [2, 3]})
    with pytest.raises(ValueError):
        graft_hybrid_tree(
            draft,
            empty_retrieval_tree(),
            k_max=10,
            matrix=m,
            root_token_id=0,
            slots=[(0, 2)],  # root parent cannot host a depth-2 slot
        )


def test_graft_hybrid_tree_into_slot_kmax_guard_raises():
    draft = {
        "node_token_ids": [1, 5, 7, 8, 9],
        "node_depths": [1, 1, 2, 2, 2],
        "parents": [-1, 0, 0, 1, 1, 2],
    }
    m = make_matrix(rows={0: [2, 3], 1: [6, 7]})
    with pytest.raises(RuntimeError):
        graft_hybrid_tree(
            draft,
            empty_retrieval_tree(),
            k_max=6,
            matrix=m,
            root_token_id=0,
            slots=[(0, 1), (1, 2)],
        )
