"""v0.3.10.1-alpha — centroid policy model (Section 3, BASELINE A/B).

A real ``PolicyModel`` implementation (not just a direction extractor).
Fits weighted class centroids and produces ``P(symbolic | h)`` via a
calibrated signed-projection mapping, so it can be compared against
the logistic and MLP routers on equal footing.

Scoring (Section 3)::

    v = μ_S - μ_L           (weighted contrastive mean)
    s(h) = h · v            (signed projection)
    p = sigmoid((s - s0) / τ_cal)

where ``s0`` is the mid-point of the signed-projection scores on the
training set (so that ``p ≈ 0.5`` at the decision boundary) and
``τ_cal`` is a temperature chosen so the projection scores map to
well-calibrated probabilities. ``τ_cal`` defaults to the standard
deviation of the training signed-projection scores.

This is BASELINE A (unweighted) or BASELINE B/D (utility-weighted)
depending on the weights supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .centroid import weighted_contrastive_mean
from .weighting import (
    InsufficientEffectiveWeight,
    ZeroWeightPolicy,
    require_effective_class_weight,
)


@dataclass
class CentroidPolicy:
    """Centroid policy model (Section 3).

    Attributes
    ----------
    vector : np.ndarray
        ``v = μ_S - μ_L`` (weighted contrastive mean direction).
    threshold : float
        ``s0`` — mid-point of training signed-projection scores.
    temperature : float
        ``τ_cal`` — calibration temperature (std of training scores).
    weight_fallback : bool
        v0.3.10.3.1 — True iff a zero-weight class triggered
        ``UNWEIGHTED_FALLBACK``. When True the estimator is NOT a true
        weighted centroid; downstream provenance must report
        ``weighted_centroid_with_unweighted_fallback``.
    fallback_classes : tuple[str, ...]
        Which class(es) triggered the fallback (for provenance).
    """

    vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    threshold: float = 0.0
    temperature: float = 1.0
    n_train_: int = 0
    n_symbolic_: int = 0
    n_llm_: int = 0
    weight_fallback: bool = False
    fallback_classes: tuple[str, ...] = ()
    degenerate_train_data: bool = False

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        """Return ``P(S | h)`` for each row of ``h``.

        Parameters
        ----------
        h : np.ndarray  shape ``[N, D]`` or ``[D]``
        """
        h = np.asarray(h, dtype=np.float32)
        if self.vector.shape[0] == 0:
            raise RuntimeError("CentroidPolicy is not fitted")
        single = h.ndim == 1
        if single:
            h = h[np.newaxis, :]
        if h.shape[1] != self.vector.shape[0]:
            raise ValueError(
                f"feature dim {h.shape[1]} != fitted dim "
                f"{self.vector.shape[0]}")
        scores = (h @ self.vector) - self.threshold
        tau = max(self.temperature, 1e-8)
        probs = 1.0 / (1.0 + np.exp(-scores / tau))
        return probs[0] if single else probs

    def predict_route(
        self,
        h: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> list[str]:
        from .abstention import choose_route
        probs = self.predict_proba(h)
        if np.isscalar(probs):
            probs = np.array([probs])
        return [choose_route(float(p), confidence_threshold).value for p in probs]

    def estimator_name(self) -> str:
        """v0.3.10.3.1 — canonical estimator name for provenance (Section 2).

        Returns ``"weighted_centroid"`` when truly weighted, or
        ``"weighted_centroid_with_unweighted_fallback"`` when a
        zero-weight class triggered ``UNWEIGHTED_FALLBACK``. Never
        report a fallback centroid as simply ``"weighted_centroid"``.
        """
        if self.weight_fallback:
            return "weighted_centroid_with_unweighted_fallback"
        return "weighted_centroid"

    # --- PolicyModel protocol (Section 3) ----------------------

    def fit(
        self,
        train_features: np.ndarray,
        train_delta_u: np.ndarray,
        train_weights: np.ndarray,
        *,
        gap_threshold: float = 0.0,
        zero_weight_policy: ZeroWeightPolicy | str = ZeroWeightPolicy.ERROR,
        eps: float = 1e-12,
    ) -> "CentroidPolicy":
        """Fit the centroid policy.

        Splits training examples into symbolic-preferred (``ΔU > gap``)
        and llm-preferred (``ΔU < -gap``); ties (``|ΔU| <= gap``) are
        excluded from the centroid computation but counted.

        Parameters
        ----------
        train_features : np.ndarray  shape ``[N, D]``
        train_delta_u : np.ndarray  shape ``[N]``
        train_weights : np.ndarray  shape ``[N]``  (>= 0)
        gap_threshold : float
            Tie band for splitting the two classes.
        zero_weight_policy : ZeroWeightPolicy | str
            v0.3.10.3.1 — behavior when a class has zero effective
            weight (Section 2). Default ``ERROR`` raises
            ``InsufficientEffectiveWeight``. ``UNWEIGHTED_FALLBACK``
            substitutes the unweighted mean, records it via
            ``weight_fallback`` / ``fallback_classes``, and the caller
            must report ``weighted_centroid_with_unweighted_fallback``.
        eps : float
            Threshold below which effective weight is considered zero.
        """
        if isinstance(zero_weight_policy, str):
            zero_weight_policy = ZeroWeightPolicy.from_str(zero_weight_policy)
        feats = np.asarray(train_features, dtype=np.float32)
        du = np.asarray(train_delta_u, dtype=np.float64)
        w = np.asarray(train_weights, dtype=np.float64)
        if feats.ndim != 2:
            raise ValueError("train_features must be [N, D]")
        if feats.shape[0] != du.shape[0] or du.shape[0] != w.shape[0]:
            raise ValueError("train_features / delta_u / weights length mismatch")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        sym_mask = du > gap_threshold
        llm_mask = du < -gap_threshold
        # If one class is empty, the routing problem is degenerate: the
        # optimal policy is a constant that always picks the non-empty
        # class. We produce a constant policy (p=1.0 for symbolic,
        # p=0.0 for LLM) rather than a meaningless 0.5. The caller
        # should detect this via n_symbolic_ / n_llm_ and the
        # degenerate_train_data flag.
        if sym_mask.sum() == 0 and llm_mask.sum() == 0:
            # Both classes empty (all ties) — truly no signal.
            self.vector = np.zeros(feats.shape[1], dtype=np.float32)
            self.threshold = 0.0
            self.temperature = 1.0
            self.n_train_ = feats.shape[0]
            self.n_symbolic_ = 0
            self.n_llm_ = 0
            self.weight_fallback = False
            self.fallback_classes = ()
            self.degenerate_train_data = True
            return self
        if llm_mask.sum() == 0:
            # All symbolic-preferred → constant p=1.0.
            # Use a large positive bias so sigmoid → 1.0.
            self.vector = np.zeros(feats.shape[1], dtype=np.float32)
            self.threshold = -100.0  # sigmoid(100/1) ≈ 1.0
            self.temperature = 1.0
            self.n_train_ = feats.shape[0]
            self.n_symbolic_ = int(sym_mask.sum())
            self.n_llm_ = 0
            self.weight_fallback = False
            self.fallback_classes = ()
            self.degenerate_train_data = True
            return self
        if sym_mask.sum() == 0:
            # All LLM-preferred → constant p=0.0.
            self.vector = np.zeros(feats.shape[1], dtype=np.float32)
            self.threshold = 100.0  # sigmoid(-100/1) ≈ 0.0
            self.temperature = 1.0
            self.n_train_ = feats.shape[0]
            self.n_symbolic_ = 0
            self.n_llm_ = int(llm_mask.sum())
            self.weight_fallback = False
            self.fallback_classes = ()
            self.degenerate_train_data = True
            return self
        sym_feats = feats[sym_mask]
        sym_w = w[sym_mask]
        llm_feats = feats[llm_mask]
        llm_w = w[llm_mask]
        # v0.3.10.3.1 — FAIL CLOSED by default if a class has zero
        # effective weight (Section 2). The old code silently fell back
        # to the unweighted mean, changing the estimator without
        # recording it in provenance. Now the behavior is explicit.
        self.weight_fallback = False
        self.fallback_classes = ()
        for name, class_w in (("symbolic", sym_w), ("llm", llm_w)):
            if float(class_w.sum()) <= eps:
                if zero_weight_policy == ZeroWeightPolicy.ERROR:
                    require_effective_class_weight(class_w, name, eps=eps)
                # UNWEIGHTED_FALLBACK: substitute unweighted mean, record it.
                self.weight_fallback = True
                self.fallback_classes = self.fallback_classes + (name,)
        if self.weight_fallback:
            if "symbolic" in self.fallback_classes:
                sym_w = np.ones_like(sym_w)
            if "llm" in self.fallback_classes:
                llm_w = np.ones_like(llm_w)
        v = weighted_contrastive_mean(
            sym_feats, sym_w, llm_feats, llm_w, normalize=False)
        self.vector = v
        # Calibrate threshold and temperature on training scores.
        scores = feats @ v
        # Mid-point: mean of the two class mean scores.
        s_sym = float((sym_feats @ v).mean())
        s_llm = float((llm_feats @ v).mean())
        self.threshold = 0.5 * (s_sym + s_llm)
        centered = scores - self.threshold
        self.temperature = float(max(centered.std(), 1e-8))
        self.n_train_ = feats.shape[0]
        self.n_symbolic_ = int(sym_mask.sum())
        self.n_llm_ = int(llm_mask.sum())
        return self

    def save(self, path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "policy_type": "centroid",
            "vector": np.asarray(self.vector, dtype=np.float32).tolist(),
            "threshold": float(self.threshold),
            "temperature": float(self.temperature),
            "n_train_": int(self.n_train_),
            "n_symbolic_": int(self.n_symbolic_),
            "n_llm_": int(self.n_llm_),
            "weight_fallback": bool(self.weight_fallback),
            "fallback_classes": list(self.fallback_classes),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CentroidPolicy":
        import json
        with open(path) as f:
            payload = json.load(f)
        if payload.get("policy_type") != "centroid":
            raise ValueError(
                f"expected policy_type='centroid', got "
                f"{payload.get('policy_type')!r}")
        model = cls()
        model.vector = np.asarray(payload["vector"], dtype=np.float32)
        model.threshold = float(payload["threshold"])
        model.temperature = float(payload["temperature"])
        model.n_train_ = int(payload.get("n_train_", 0))
        model.n_symbolic_ = int(payload.get("n_symbolic_", 0))
        model.n_llm_ = int(payload.get("n_llm_", 0))
        model.weight_fallback = bool(payload.get("weight_fallback", False))
        model.fallback_classes = tuple(payload.get("fallback_classes", []))
        return model


def train_centroid_policy(
    train_features: np.ndarray,
    train_delta_u: np.ndarray,
    train_weights: np.ndarray,
    *,
    gap_threshold: float = 0.0,
    zero_weight_policy: ZeroWeightPolicy | str = ZeroWeightPolicy.ERROR,
    eps: float = 1e-12,
) -> CentroidPolicy:
    """Convenience: construct + fit a :class:`CentroidPolicy`."""
    model = CentroidPolicy()
    return model.fit(
        train_features, train_delta_u, train_weights,
        gap_threshold=gap_threshold,
        zero_weight_policy=zero_weight_policy,
        eps=eps)


__all__ = ["CentroidPolicy", "train_centroid_policy"]
