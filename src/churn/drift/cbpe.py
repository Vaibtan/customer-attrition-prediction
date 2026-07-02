"""CBPE -- Confidence-Based Performance Estimation of ROC-AUC on UNLABELED data.

When labels arrive late (the churn label is only known 90 days later), we still want an estimate of
how the deployed model is doing *now*. CBPE estimates ROC-AUC from the model's own calibrated
scores: treat each instance as a soft positive (weight ``p``) and soft negative (weight ``1-p``) and
compute the expected Mann-Whitney statistic -- O(n log n), no labels needed.

**Assumption:** the scores are calibrated AND the score -> outcome relationship is stable.
**Failure criterion (stated, not hidden):** under *concept drift* the feature -> label relationship
changes but the scores do not, so CBPE keeps reporting the old performance -- it is **blind to
concept drift by construction**. Always report the estimate WITH its bootstrap band and this caveat;
``concept_drift_suspected`` flags when a known true value falls outside the band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CAVEAT = "CBPE assumes calibration + no concept drift; it is blind to concept drift by design."


def estimate_auc(proba) -> float:
    """Expected ROC-AUC from calibrated scores alone (soft-label Mann-Whitney, O(n log n))."""
    p = np.asarray(proba, dtype="float64")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    pos_mass = ps.sum()
    neg_mass = (1.0 - ps).sum()
    if pos_mass <= 0 or neg_mass <= 0:
        return 0.5
    neg_below = np.cumsum(1.0 - ps) - (1.0 - ps)  # exclusive prefix: neg weight at strictly lower s
    numerator = float(np.sum(ps * neg_below))
    return numerator / (pos_mass * neg_mass)


@dataclass(frozen=True)
class CBPEResult:
    estimate: float
    lo: float
    hi: float
    caveat: str = _CAVEAT

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


def estimate(proba, n_rounds: int = 500, seed: int = 42, alpha: float = 0.05) -> CBPEResult:
    """CBPE ROC-AUC point estimate + a percentile bootstrap band over the scored instances."""
    p = np.asarray(proba, dtype="float64")
    point = estimate_auc(p)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(p), size=(n_rounds, len(p)))
    boot = np.array([estimate_auc(p[idx]) for idx in draws])
    return CBPEResult(
        estimate=point,
        lo=float(np.quantile(boot, alpha / 2)),
        hi=float(np.quantile(boot, 1 - alpha / 2)),
    )


def concept_drift_suspected(result: CBPEResult, realized_auc: float) -> bool:
    """True when a realized (labeled) AUC falls outside the CBPE band -- the estimator was blind."""
    return not result.contains(realized_auc)
