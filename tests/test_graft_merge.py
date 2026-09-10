"""Integration-style tests for the prune-then-graft fusion step.

``eval_dartree.build_dartree_supertree`` performs the real fusion; these tests
re-implement that merge over the exact same building blocks
(``select_topb_prefix_tree`` + ``RetrievalTemplate.materialize`` + index
remapping) and assert the invariants the real code relies on: budget
conservation, topological order, prefix closure, retrieval nodes hanging off
the root instead of draft nodes, and token/rank consistency with the
adjacency matrix.

Run from the repository root on a machine with torch + transformers (CPU is
enough):

    python -m pytest tests/test_graft_merge.py -v
"""

import torch
import pytest

from utils.retrieval import (
    GraftConfig,
    RetrievalAdjacencyMatrix,
    RetrievalTemplate,
    make_decaying_depth_counts,
)
from eval_dartree import parse_float_list, parse_int_list, select_topb_prefix_tree


# --- parse helpers ---------------------------------------------------------


def test_parse_int_list():
    assert parse_int_list("0,1,5") == [0, 1, 5]
    assert parse_int_list(" 0 , 1 ") == [0, 1]
    assert parse_int_list("") == []


def test_parse_float_list():
    assert parse_float_list("0.35,0.25,0.15") == [0.35, 0.25, 0.15]
    assert parse_float_list("0.14, 0.14, 0.49") == [0.14, 0.14, 0.49]
    assert parse_float_list("") == []


# --- select_topb_prefix_tree -----------------------------------------------


def test_topb_basic():
    parents = [-1, 0, 1, 1, 2]
    scores = [0.0, -0.1, -0.2, -0.25, -0.3]
    kept = select_topb_prefix_tree(parents, scores, 3)
    # scores by node: 1=-0.1, 2=-0.2, 3=-0.25, 4=-0.3 -> top-3 = nodes 1, 2, 3
    assert kept == [1, 2, 3]


def test_topb_prefix_closed_on_random_monotone_trees():
    torch.manual_seed(0)
    for _ in range(20):
        parents, scores = [-1], [0.0]
        frontier = [0]
        for _depth in range(4):
            next_frontier = []
            for parent in frontier:
                for _child in range(3):
                    parents.append(parent)
                    scores.append(
                        scores[parent] - abs(torch.randn(1).item()) - 0.01
                    )
                    next_frontier.append(len(parents) - 1)
            frontier = next_frontier
        node_count = len(parents) - 1
        for budget in (1, 4, 9, node_count):
            kept = select_topb_prefix_tree(parents, scores, budget)
            assert len(kept) == budget
            assert kept == sorted(kept)
            kept_set = {0, *kept}
            for node in kept:
                assert parents[node] in kept_set, (node, parents[node], kept)


def test_topb_budget_edges():
    parents = [-1, 0, 0, 1]
    scores = [0.0, -0.1, -0.2, -0.3]
    assert select_topb_prefix_tree(parents, scores, 1) == [1]
    assert select_topb_prefix_tree(parents, scores, 3) == [1, 2, 3]


def test_topb_invalid_parent_raises():
    parents = [-1, 2, 0]  # node 1's parent 2 >= node 1
    scores = [0.0, -0.1, -0.2]
    with pytest.raises(ValueError):
        select_topb_prefix_tree(parents, scores, 2)


def test_topb_non_monotone_raises():
    parents = [-1, 0, 1]
    scores = [0.0, -0.1, -0.05]  # child score higher than its parent
    with pytest.raises(ValueError):
        select_topb_prefix_tree(parents, scores, 2)


def test_topb_budget_out_of_range_raises():
    parents = [-1, 0, 1]
    scores = [0.0, -0.1, -0.2]
    with pytest.raises(ValueError):
        select_topb_prefix_tree(parents, scores, 5)
    with pytest.raises(ValueError):
        select_topb_prefix_tree(parents, scores, -1)


# --- graft fusion (mirrors eval_dartree.build_dartree_supertree) -----------


def _fuse_draft_and_retrieval(
    *,
    draft_parents,
    draft_scores,
    draft_tokens,
    draft_depths,
    root_token_id,
    budget,
    stage,
    config,
    matrix,
    depth_limit,
):
    """Re-implements the graft merge block of ``build_dartree_supertree``."""
    node_count = len(draft_tokens)
    b_draft = max(1, min(node_count, config.draft_budget(stage, budget)))
    kept = select_topb_prefix_tree(draft_parents, draft_scores, b_draft)
    old_to_new = {0: 0}
    old_to_new.update({old: new for new, old in enumerate(kept, start=1)})
    fused_parents = [-1] + [old_to_new[draft_parents[node]] for node in kept]
    fused_tokens = [draft_tokens[node] for node in kept]
    fused_depths = [draft_depths[node] for node in kept]

    retrieved_valid = 0
    b_ret = int(budget) - b_draft
    if b_ret > 0:
        counts = make_decaying_depth_counts(
            b_ret, depth_limit, min_width=config.min_template_width
        )
        template = RetrievalTemplate(counts, matrix.k)
        t_tokens, t_parents, _ranks, t_valid = template.materialize(
            root_token_id, matrix
        )
        global_map = {0: 0}
        next_global = b_draft + 1
        for template_index in range(template.node_count):
            if not t_valid[template_index]:
                global_map[template_index + 1] = None
                continue
            global_id = next_global
            next_global += 1
            global_map[template_index + 1] = global_id
            fused_parents.append(global_map[t_parents[template_index]])
            fused_tokens.append(t_tokens[template_index])
            fused_depths.append(template.node_depths[template_index])
            retrieved_valid += 1
    return fused_parents, fused_tokens, fused_depths, b_draft, retrieved_valid


def _fill_matrix(matrix, rows):
    """Fill rows via one-hot-ish logits so argtop_k == the given successors."""
    vocab = matrix.vocab_size
    for token, successors in rows.items():
        logits = torch.zeros(1, vocab)
        for rank, successor in enumerate(successors):
            logits[0, successor] = float(matrix.k - rank)
        matrix.update_from_logits(torch.tensor([token]), logits)


# a 2-level draft supertree: nodes 1..3 (depth 1) and 4, 5 (depth 2)
_DRAFT_PARENTS = [-1, 0, 0, 0, 1, 2]
_DRAFT_SCORES = [0.0, -0.1, -0.2, -0.3, -0.25, -0.35]
_DRAFT_TOKENS = [100, 101, 102, 110, 111]
_DRAFT_DEPTHS = [1, 1, 1, 2, 2]


def test_fusion_invariants_full_matrix():
    """stage-1 prune + a fully usable matrix fills the budget exactly and
    keeps every fused-tree invariant."""
    budget, depth_limit = 10, 3
    config = GraftConfig()
    matrix = RetrievalAdjacencyMatrix(300, 3, "cpu")
    _fill_matrix(
        matrix,
        {
            50: [200, 201, 202],   # root row
            200: [210, 211, 212],  # retrieved node 1's row
            201: [220, 221, 222],  # retrieved node 2's row
            211: [230, 231, 232],  # retrieved chain row (token 211)
        },
    )
    fused_parents, fused_tokens, fused_depths, b_draft, n_ret = (
        _fuse_draft_and_retrieval(
            draft_parents=_DRAFT_PARENTS,
            draft_scores=_DRAFT_SCORES,
            draft_tokens=_DRAFT_TOKENS,
            draft_depths=_DRAFT_DEPTHS,
            root_token_id=50,
            budget=budget,
            stage=1,
            config=config,
            matrix=matrix,
            depth_limit=depth_limit,
        )
    )
    total = len(fused_tokens)
    assert b_draft == 4  # min(5, round(0.4 * 10)) = 4
    assert n_ret == total - 4
    assert total == budget  # 4 draft + 6 retrieved = 10

    # draft part preserved in kept order (nodes 1, 2, 3, 4 by score)
    assert fused_tokens[:4] == [100, 101, 102, 110]
    assert fused_depths[:4] == [1, 1, 1, 2]
    assert fused_parents[0] == -1
    assert fused_parents[1:5] == [0, 0, 0, 1]

    # whole fused tree: topological order + prefix closure
    for index in range(1, total + 1):
        assert 0 <= fused_parents[index] < index, (index, fused_parents[index])

    # retrieval nodes hang off the root or earlier retrieval nodes
    for index in range(5, total + 1):
        parent = fused_parents[index]
        assert parent == 0 or parent >= 5, (index, parent)

    # retrieval slice: tokens follow the template walk over the matrix and
    # parents map through the template -> global index remapping
    assert fused_tokens[4:] == [200, 201, 202, 211, 221, 230]
    assert fused_parents[4:] == [0, 0, 0, 5, 6, 8]


def test_fusion_cold_matrix_keeps_draft_only():
    """Cold-start matrix: every retrieval node drops (prefix-closed), leaving
    exactly the retained draft tree."""
    budget, depth_limit = 10, 3
    config = GraftConfig()
    matrix = RetrievalAdjacencyMatrix(300, 3, "cpu")  # nothing filled
    fused_parents, fused_tokens, fused_depths, b_draft, n_ret = (
        _fuse_draft_and_retrieval(
            draft_parents=_DRAFT_PARENTS,
            draft_scores=_DRAFT_SCORES,
            draft_tokens=_DRAFT_TOKENS,
            draft_depths=_DRAFT_DEPTHS,
            root_token_id=50,
            budget=budget,
            stage=1,
            config=config,
            matrix=matrix,
            depth_limit=depth_limit,
        )
    )
    assert b_draft == 4
    assert n_ret == 0
    assert fused_tokens == [100, 101, 102, 110]
    assert fused_depths == [1, 1, 1, 2]
    assert fused_parents == [-1, 0, 0, 0, 1]


def test_fusion_root_stage_budget_bounds():
    """stage-0 prune with a small budget exercises the max(1, ...) floor and
    partial retrieval validity."""
    budget, depth_limit = 8, 3
    config = GraftConfig()
    matrix = RetrievalAdjacencyMatrix(300, 3, "cpu")
    _fill_matrix(matrix, {50: [200, 201, 202]})  # only the root row
    fused_parents, fused_tokens, fused_depths, b_draft, n_ret = (
        _fuse_draft_and_retrieval(
            draft_parents=_DRAFT_PARENTS,
            draft_scores=_DRAFT_SCORES,
            draft_tokens=_DRAFT_TOKENS,
            draft_depths=_DRAFT_DEPTHS,
            root_token_id=50,
            budget=budget,
            stage=0,
            config=config,
            matrix=matrix,
            depth_limit=depth_limit,
        )
    )
    assert b_draft == 1  # max(1, min(5, round(0.13 * 8))) = 1
    assert fused_tokens[0] == 100
    assert fused_depths[0] == 1
    assert n_ret == len(fused_tokens) - 1
    assert len(fused_tokens) <= budget
    for index in range(1, len(fused_tokens) + 1):
        assert 0 <= fused_parents[index] < index, (index, fused_parents[index])
