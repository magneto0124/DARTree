"""Unit tests for ``GraftConfig`` (``utils/retrieval.py``).

Covers the prune-then-graft configuration: validation of checkpoints /
thresholds / stage fractions, checkpoint lookup, and the stage-adaptive
draft/retrieval budget split (paper Eq. 13).

Run from the repository root on a machine with torch (CPU is enough):

    python -m pytest tests/test_graft_config.py -v
"""

import pytest

from utils.retrieval import GraftConfig


def test_defaults():
    config = GraftConfig()
    assert config.k == 9
    assert config.checkpoints == (0, 1, 5)
    assert config.thresholds == (0.35, 0.25, 0.15)
    assert config.stage_draft_fractions == (0.13, 0.4, 0.67)
    assert config.min_template_width == 1
    assert config.init_from_draft_logits is True


def test_checkpoint_index():
    config = GraftConfig()
    assert config.checkpoint_index(0) == 0
    assert config.checkpoint_index(1) == 1
    assert config.checkpoint_index(5) == 2
    assert config.checkpoint_index(2) == -1
    assert config.checkpoint_index(4) == -1


def test_paper_budget_example_budget_60():
    """The paper's 60-node example: stages keep 8/24/40 draft nodes and give
    52/36/20 slots to retrieval."""
    config = GraftConfig()
    assert config.draft_budget(0, 60) == 8
    assert config.draft_budget(1, 60) == 24
    assert config.draft_budget(2, 60) == 40
    assert config.retrieval_budget(0, 60) == 52
    assert config.retrieval_budget(1, 60) == 36
    assert config.retrieval_budget(2, 60) == 20


def test_budget_rounding_at_budget_64():
    """DARTree's default --tree-budget 64."""
    config = GraftConfig()
    assert config.draft_budget(0, 64) == 8  # round(0.13 * 64) = round(8.32)
    assert config.draft_budget(1, 64) == 26  # round(25.6)
    assert config.draft_budget(2, 64) == 43  # round(42.88)
    assert config.retrieval_budget(2, 64) == 21


def test_draft_budget_bounds():
    config = GraftConfig()
    assert config.draft_budget(0, 1) == 1  # at least 1
    assert config.draft_budget(1, 2) == 1  # max(1, min(2, round(0.8))) = 1
    assert config.retrieval_budget(0, 1) == 0


def test_retrieval_budget_complement():
    config = GraftConfig()
    for stage in range(len(config.checkpoints)):
        for budget in (8, 16, 60, 64, 128):
            assert config.draft_budget(stage, budget) + config.retrieval_budget(
                stage, budget
            ) == budget


def test_custom_config():
    config = GraftConfig(
        k=12,
        checkpoints=(0, 2, 4),
        thresholds=(0.1, 0.2, 0.3),
        stage_draft_fractions=(0.2, 0.5, 0.8),
    )
    assert config.checkpoint_index(0) == 0
    assert config.checkpoint_index(2) == 1
    assert config.checkpoint_index(4) == 2
    assert config.draft_budget(2, 100) == 80


# --- validation ------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        GraftConfig(checkpoints=(0, 1), thresholds=(0.3,))
    with pytest.raises(ValueError):
        GraftConfig(checkpoints=(0, 1), stage_draft_fractions=(0.1,))


def test_duplicate_checkpoints_raise():
    with pytest.raises(ValueError):
        GraftConfig(checkpoints=(0, 0, 5))


def test_negative_checkpoint_raises():
    with pytest.raises(ValueError):
        GraftConfig(checkpoints=(-1, 1, 5))


def test_nonpositive_threshold_raises():
    with pytest.raises(ValueError):
        GraftConfig(thresholds=(0.0, 0.2, 0.1))
    with pytest.raises(ValueError):
        GraftConfig(thresholds=(-0.1, 0.2, 0.1))


def test_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        GraftConfig(stage_draft_fractions=(0.0, 0.4, 0.67))
    with pytest.raises(ValueError):
        GraftConfig(stage_draft_fractions=(1.0, 0.4, 0.67))
    with pytest.raises(ValueError):
        GraftConfig(stage_draft_fractions=(1.5, 0.4, 0.67))
