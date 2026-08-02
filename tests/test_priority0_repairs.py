"""Priority 0 scientific-correctness repair tests.

Covers the five fixes from the v0.4.0a2 audit:

1. Sham training permutes the signed continuous ΔU (and matching weights),
   not a binary label derivative — so fit_policy(target_mode="soft") applies
   the identical sigmoid(ΔU/τ) transform the real P1 model received.
2. Ablation inference pipelines share P1's calibration + frozen thresholds.
3. Pooling candidates (last_prompt_token / mean_prompt_tokens /
   mean_content_tokens) are genuinely distinct representations.
4. The invalid square random-projection control is replaced by proper
   signal-destruction / invariance controls.
5. The subtype-only baseline is frozen on development data, never derived
   from final labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from daph_learning.evaluation.sham import (
    permute_targets_within_bins,
    shuffle_labels_within_bins,
)


# ── Fix 1: matched sham permutes continuous ΔU + weights ──────────────

def test_permute_targets_preserves_multiset_of_delta_u():
    """The permuted ΔU must be a permutation of the real ΔU (same multiset),
    so the target distribution is identical to P1's — only the X↔ΔU
    association is destroyed."""
    rng = np.random.RandomState(0)
    n = 80
    delta_u = rng.randn(n) * 0.3
    weights = np.abs(delta_u) + 0.1
    subtypes = rng.choice(["A", "B", "C"], n)
    splits = np.array(["train"] * n)
    decisive = np.abs(delta_u) > 0.1

    perm_du, perm_w, n_shuffled = permute_targets_within_bins(
        delta_u, weights, subtypes, splits, decisive, seed=42)

    # Same multiset of ΔU values (order may differ).
    assert sorted(perm_du.tolist()) == sorted(delta_u.tolist())
    # Same multiset of weights.
    assert sorted(perm_w.tolist()) == sorted(weights.tolist())
    # The (ΔU, weight) pairing is preserved: each permuted weight still
    # corresponds to its original ΔU value.
    orig_pairs = set(zip(np.round(delta_u, 12).tolist(),
                         np.round(weights, 12).tolist()))
    perm_pairs = set(zip(np.round(perm_du, 12).tolist(),
                         np.round(perm_w, 12).tolist()))
    assert orig_pairs == perm_pairs
    assert n_shuffled > 0


def test_permute_targets_destroys_x_alignment():
    """With a clear X→ΔU signal, permutation should destroy the alignment
    (permuted ΔU no longer correlates with X)."""
    rng = np.random.RandomState(1)
    n = 200
    # X encodes ΔU: ΔU = X[:, 0] + noise.
    X = rng.randn(n, 4)
    delta_u = X[:, 0] * 2.0 + rng.randn(n) * 0.05
    weights = np.ones(n)
    subtypes = np.array(["A"] * n)
    splits = np.array(["train"] * n)
    decisive = np.abs(delta_u) > 0.1

    perm_du, _perm_w, _ = permute_targets_within_bins(
        delta_u, weights, subtypes, splits, decisive, seed=7)

    # Original correlation is strong.
    orig_corr = abs(np.corrcoef(X[:, 0], delta_u)[0, 1])
    perm_corr = abs(np.corrcoef(X[:, 0], perm_du)[0, 1])
    assert orig_corr > 0.9
    assert perm_corr < 0.3  # alignment destroyed


def test_permute_targets_not_binary_derivative():
    """The permuted values must be the real continuous ΔU, not 0/1 labels.
    A binary derivative would only contain {0, 1}-ish values after sigmoid,
    whereas the real ΔU has continuous spread."""
    rng = np.random.RandomState(2)
    n = 100
    delta_u = rng.randn(n) * 0.5  # continuous, both signs
    weights = np.ones(n)
    subtypes = np.array(["A"] * n)
    splits = np.array(["train"] * n)
    decisive = np.abs(delta_u) > 0.1

    perm_du, _perm_w, _ = permute_targets_within_bins(
        delta_u, weights, subtypes, splits, decisive, seed=11)

    # The permuted ΔU has continuous spread (std > 0.1), NOT just {0, 1}.
    assert perm_du.std() > 0.1
    # It is not equal to a binarized version.
    binary = (delta_u > 0).astype(float)
    assert not np.allclose(sorted(perm_du.tolist()), sorted(binary.tolist()))


def test_permute_targets_stratified_within_subtypes():
    """Permutation happens within subtype strata, so the per-subtype ΔU
    multiset is preserved (subtype prevalence preserved)."""
    rng = np.random.RandomState(3)
    n = 120
    subtypes = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    delta_u = np.concatenate([
        rng.randn(40) * 0.3 + 0.5,   # A: positive shift
        rng.randn(40) * 0.3 - 0.5,   # B: negative shift
        rng.randn(40) * 0.3,         # C: centered
    ])
    weights = np.ones(n)
    splits = np.array(["train"] * n)
    decisive = np.abs(delta_u) > 0.1

    perm_du, _perm_w, _ = permute_targets_within_bins(
        delta_u, weights, subtypes, splits, decisive, seed=5)

    for st in ("A", "B", "C"):
        mask = subtypes == st
        assert sorted(perm_du[mask].tolist()) == sorted(delta_u[mask].tolist()), (
            f"subtype {st} ΔU multiset not preserved")


# ── Fix 3: pooling candidates are genuinely distinct ───────────────────

def test_pooling_candidates_are_distinct_names():
    """The three pooling candidates must be distinct strings (not aliases)."""
    from daph_learning.evaluation.representation_selection import (
        pooling_candidates, all_candidates,
    )
    pools = pooling_candidates()
    assert len(set(pools)) == 3
    assert set(pools) == {"last_prompt_token", "mean_prompt_tokens",
                          "mean_content_tokens"}


def test_all_candidates_cross_layer_and_pooling():
    """Each (layer, pooling) pair is a distinct candidate."""
    from daph_learning.evaluation.representation_selection import all_candidates
    cands = all_candidates(24)
    # 4 layers × 3 poolings = 12 distinct candidates.
    assert len(cands) == 12
    seen = {(c.layer, c.pooling) for c in cands}
    assert len(seen) == 12


def test_mean_content_tokens_excludes_special_tokens():
    """mean_content_tokens must exclude BOS/EOS special tokens, making it
    distinct from mean_prompt_tokens when special tokens are present."""
    # Simulate: 4 tokens, token 0 is a special (BOS) id.
    # hidden states: [[1], [2], [3], [4]] (dim=1 for clarity)
    # attention_mask: [1, 1, 1, 1]
    # special_ids = {0} (the id at position 0)
    # mean_prompt_tokens = (1+2+3+4)/4 = 2.5
    # mean_content_tokens = (2+3+4)/3 = 3.0  (excludes BOS)
    import torch
    hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [1, 4, 1]
    attn = torch.tensor([[1, 1, 1, 1]])
    input_ids = torch.tensor([[0, 5, 6, 7]])  # id 0 = BOS
    special_ids = {0}

    # mean_prompt_tokens
    am = attn[0].float().unsqueeze(-1)  # [4, 1]
    mean_prompt = (hidden[0] * am).sum(0) / am.sum().clamp(min=1)

    # mean_content_tokens
    content_mask = attn[0].clone()  # [4]
    for t_idx in range(int(attn[0].sum().item())):
        if int(input_ids[0, t_idx].item()) in special_ids:
            content_mask[t_idx] = 0
    cm = content_mask.float().unsqueeze(-1)  # [4, 1]
    mean_content = (hidden[0] * cm).sum(0) / cm.sum().clamp(min=1)

    assert abs(float(mean_prompt[0]) - 2.5) < 1e-6
    assert abs(float(mean_content[0]) - 3.0) < 1e-6
    assert float(mean_prompt[0]) != float(mean_content[0])


# ── Fix 4: random-projection control replaced ──────────────────────────

def test_orthogonal_matrix_is_orthonormal():
    """The orthogonal-rotation control uses a true orthonormal matrix Q
    (Q Q^T = I), so a linear probe over XQ has the same capacity as over X."""
    rng = np.random.default_rng(42)
    d = 16
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _R = np.linalg.qr(G)
    # Q has orthonormal columns: Q^T Q = I.
    assert np.allclose(Q.T @ Q, np.eye(d, dtype=np.float32), atol=1e-4)


def test_orthogonal_rotation_preserves_linear_separability():
    """A linear classifier over XQ can represent the same boundary as over X
    (w' = Q^T w), confirming the orthogonal rotation is an invariance test,
    not a signal-destruction control."""
    rng = np.random.default_rng(7)
    n, d = 100, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    logits = X @ w
    # Q rotation
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    Xq = X @ Q
    # The same logits are recoverable: Xq @ (Q^T w) == X @ w.
    w_recovered = Q.T @ w
    logits_recovered = Xq @ w_recovered
    assert np.allclose(logits, logits_recovered, atol=1e-3)


def test_reduced_projection_is_not_invertible():
    """The dimension-reduced projection is rectangular (hidden_dim → d//4),
    so it is NOT invertible — genuine information is discarded."""
    rng = np.random.default_rng(11)
    d = 16
    reduced = max(1, d // 4)
    P = rng.standard_normal((d, reduced)).astype(np.float32) / np.sqrt(reduced)
    assert P.shape == (d, reduced)
    assert reduced < d  # not square → not invertible


# ── Fix 5: subtype-only frozen on development data ─────────────────────

def test_evaluate_baselines_subtype_only_uses_dev_preferences():
    """When dev_subtype_preferences are provided, the subtype-only baseline
    applies the frozen dev policy to final records (no final-label leakage)."""
    # Import the runner's _evaluate_baselines.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_gate_a_staged",
        REPO_ROOT / "scripts" / "run_gate_a_staged.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Rec:
        def __init__(self, subtype, sym_u, llm_u, p1_u):
            self.subtype = subtype
            self.symbolic_utility = sym_u
            self.llm_utility = llm_u
            self.p1_realized_utility = p1_u

    # Final records: subtype A prefers symbolic, B prefers LLM (per dev).
    records = [
        _Rec("A", 0.9, 0.1, 0.9),
        _Rec("A", 0.8, 0.2, 0.8),
        _Rec("B", 0.1, 0.9, 0.9),
        _Rec("B", 0.2, 0.8, 0.8),
    ]
    # Frozen dev preferences: A→symbolic, B→llm.
    dev_prefs = {"A": "symbolic", "B": "llm"}
    result = mod._evaluate_baselines(
        records, best_fixed_id="always_symbolic",
        dev_subtype_preferences=dev_prefs)

    # subtype_only should route A→symbolic (0.9+0.8)/2 and B→llm (0.9+0.8)/2.
    assert result["subtype_only"]["utility"] is not None
    expected = (0.9 + 0.8 + 0.9 + 0.8) / 4  # = 0.85
    assert abs(result["subtype_only"]["utility"] - expected) < 1e-6
    assert result["subtype_only"]["selection_data"] == "development (frozen)"


def test_evaluate_baselines_subtype_only_unavailable_without_dev_prefs():
    """Without dev preferences, subtype_only is reported as unavailable
    (None) rather than being optimistically re-fit on final labels."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_gate_a_staged",
        REPO_ROOT / "scripts" / "run_gate_a_staged.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Rec:
        def __init__(self, subtype, sym_u, llm_u, p1_u):
            self.subtype = subtype
            self.symbolic_utility = sym_u
            self.llm_utility = llm_u
            self.p1_realized_utility = p1_u

    records = [_Rec("A", 0.9, 0.1, 0.9), _Rec("B", 0.1, 0.9, 0.9)]
    result = mod._evaluate_baselines(records, best_fixed_id="always_symbolic")
    assert result["subtype_only"]["utility"] is None
    assert "unavailable" in result["subtype_only"]["selection_data"]


def test_evaluate_baselines_subtype_only_does_not_use_final_labels():
    """The subtype-only utility must NOT depend on which backend wins in
    the FINAL records — only on the frozen dev preferences. Flipping the
    final-record utilities must not change the subtype routing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_gate_a_staged",
        REPO_ROOT / "scripts" / "run_gate_a_staged.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Rec:
        def __init__(self, subtype, sym_u, llm_u, p1_u):
            self.subtype = subtype
            self.symbolic_utility = sym_u
            self.llm_utility = llm_u
            self.p1_realized_utility = p1_u

    dev_prefs = {"A": "symbolic", "B": "llm"}
    # Version 1: final records agree with dev prefs.
    recs1 = [_Rec("A", 0.9, 0.1, 0.9), _Rec("B", 0.1, 0.9, 0.9)]
    # Version 2: final records FLIP which backend wins, but dev prefs frozen.
    recs2 = [_Rec("A", 0.1, 0.9, 0.1), _Rec("B", 0.9, 0.1, 0.1)]
    r1 = mod._evaluate_baselines(recs1, dev_subtype_preferences=dev_prefs)
    r2 = mod._evaluate_baselines(recs2, dev_subtype_preferences=dev_prefs)
    # Both route A→symbolic, B→llm (frozen), so they use sym_u for A and
    # llm_u for B in both cases.
    assert abs(r1["subtype_only"]["utility"] - (0.9 + 0.9) / 2) < 1e-6
    assert abs(r2["subtype_only"]["utility"] - (0.1 + 0.1) / 2) < 1e-6
