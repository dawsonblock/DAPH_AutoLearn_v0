"""Section 12 — representation selection tests."""

from __future__ import annotations

import pytest

from daph_learning.evaluation.representation_selection import (
    RepresentationCandidate,
    RepresentationSelection,
    all_candidates,
    layer_candidates,
    pooling_candidates,
    select_representation,
    CandidateResult,
)


def test_layer_candidates_24_layers():
    cands = layer_candidates(24)
    # 0.25*23=6, 0.50*23=12, 0.75*23=17, last=23
    assert cands == [6, 12, 17, 23]


def test_layer_candidates_small_model():
    cands = layer_candidates(4)
    # 0.25*3=1, 0.50*3=2, 0.75*3=2, last=3
    assert 3 in cands
    assert len(cands) >= 2


def test_pooling_candidates():
    pools = pooling_candidates()
    assert "last_prompt_token" in pools
    assert "mean_prompt_tokens" in pools
    assert "mean_content_tokens" in pools


def test_all_candidates_24_layers():
    cands = all_candidates(24)
    assert len(cands) == 4 * 3  # 4 layers × 3 pooling


def test_select_representation_picks_best():
    results = [
        CandidateResult(
            candidate=RepresentationCandidate(layer=6, pooling="mean_prompt_tokens"),
            dev_objective=0.3, mean_utility=0.5, mean_regret=0.1,
            p_symbolic=0.4, abstention_rate=0.1, n_tasks=100),
        CandidateResult(
            candidate=RepresentationCandidate(layer=12, pooling="last_prompt_token"),
            dev_objective=0.5, mean_utility=0.7, mean_regret=0.05,
            p_symbolic=0.5, abstention_rate=0.05, n_tasks=100),
        CandidateResult(
            candidate=RepresentationCandidate(layer=23, pooling="mean_content_tokens"),
            dev_objective=0.4, mean_utility=0.6, mean_regret=0.08,
            p_symbolic=0.45, abstention_rate=0.08, n_tasks=100),
    ]
    sel = select_representation(results, n_layers=24)
    assert sel.selected.layer == 12
    assert sel.selected.pooling == "last_prompt_token"
    assert len(sel.all_results) == 3


def test_select_representation_tie_breaking():
    """Ties broken by lowest layer then alphabetical pooling."""
    results = [
        CandidateResult(
            candidate=RepresentationCandidate(layer=23, pooling="last_prompt_token"),
            dev_objective=0.5, mean_utility=0.7, mean_regret=0.05,
            p_symbolic=0.5, abstention_rate=0.05, n_tasks=100),
        CandidateResult(
            candidate=RepresentationCandidate(layer=6, pooling="mean_content_tokens"),
            dev_objective=0.5, mean_utility=0.7, mean_regret=0.05,
            p_symbolic=0.5, abstention_rate=0.05, n_tasks=100),
    ]
    sel = select_representation(results, n_layers=24)
    # Same objective → lowest layer wins (6 < 23)
    assert sel.selected.layer == 6


def test_select_representation_empty_raises():
    with pytest.raises(ValueError):
        select_representation([])


def test_selection_hash_is_deterministic():
    results = [
        CandidateResult(
            candidate=RepresentationCandidate(layer=12, pooling="last_prompt_token"),
            dev_objective=0.5, mean_utility=0.7, mean_regret=0.05,
            p_symbolic=0.5, abstention_rate=0.05, n_tasks=100),
    ]
    sel = select_representation(results, n_layers=24)
    assert sel.selection_hash == sel.selection_hash
    assert len(sel.selection_hash) == 64


def test_to_dict_serializable():
    results = [
        CandidateResult(
            candidate=RepresentationCandidate(layer=12, pooling="last_prompt_token"),
            dev_objective=0.5, mean_utility=0.7, mean_regret=0.05,
            p_symbolic=0.5, abstention_rate=0.05, n_tasks=100),
    ]
    sel = select_representation(results, n_layers=24)
    d = sel.to_dict()
    assert "selected" in d
    assert "all_results" in d
    assert d["selected"]["layer"] == 12
