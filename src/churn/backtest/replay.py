"""Backtest replay harness -- drive a drifting timeline, record estimated vs true performance.

The centerpiece narrative (ENHANCEMENT_PLAN.md Sec 7). A champion is trained on a reference regime,
then the world **rotates its decision boundary** (concept drift): the champion's *true* ROC-AUC
(delayed labels) decays, but its scores stay confident so **CBPE keeps reporting the old, optimistic
number** -- and the covariate detectors stay silent (the marginals never move). Only a delayed-label
performance monitor catches it; when it does, a retrain on the freshly-labeled window recovers
performance. The returned per-step frame is exactly what the centerpiece chart plots: estimated vs
true, drift/retrain/recovery markers.

This is a synthetic-domain systems demonstration of the monitoring + lifecycle machinery, not a
real-world performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from churn.drift import cbpe
from churn.drift import detectors as DET

_FEATURES = ["f0", "f1"]


@dataclass(frozen=True)
class BacktestConfig:
    n_steps: int = 14
    drift_start: int = 4
    ramp_steps: int = 4  # steps over which the boundary rotates to its max angle
    max_angle: float = 1.25  # radians the decision boundary rotates (concept drift magnitude)
    window_n: int = 2500
    retrain_threshold: float = 0.72  # delayed-label AUC below this triggers a retrain
    seed: int = 0


def _angle(step: int, cfg: BacktestConfig) -> float:
    if step < cfg.drift_start:
        return 0.0
    progressed = min(1.0, (step - cfg.drift_start) / max(cfg.ramp_steps, 1))
    return cfg.max_angle * progressed


def _window(rng: np.random.Generator, n: int, angle: float) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 1.0, (n, 2))
    boundary = np.cos(angle) * x[:, 0] + np.sin(angle) * x[:, 1]
    y = (boundary + rng.normal(0.0, 0.5, n) > 0).astype(int)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(x, y)


def run_backtest(cfg: BacktestConfig | None = None) -> pd.DataFrame:
    """Replay the drifting timeline; return a per-step frame of estimated vs true perf + events."""
    cfg = cfg or BacktestConfig()
    rng = np.random.default_rng(cfg.seed)
    x_ref, y_ref = _window(rng, cfg.window_n, angle=0.0)
    reference = pd.DataFrame(x_ref, columns=_FEATURES)
    champion = _fit(x_ref, y_ref)

    rows = []
    steps_since_retrain = 99
    for step in range(cfg.n_steps):
        x, y = _window(rng, cfg.window_n, _angle(step, cfg))
        proba = champion.predict_proba(x)[:, 1]
        true_auc = float(roc_auc_score(y, proba))
        band = cbpe.estimate(proba, n_rounds=100, seed=cfg.seed + step)
        # Covariate detectors see the (unchanged) marginals -> silent under pure concept drift.
        covariate = DET.detect_drift(
            reference, pd.DataFrame(x, columns=_FEATURES), _FEATURES, []
        ).covariate_shift

        retrained = False
        # Delayed-label performance monitor: retrain on the freshly-labeled window when true perf
        # has decayed past the threshold (and we did not just retrain).
        if true_auc < cfg.retrain_threshold and steps_since_retrain >= 2:
            champion = _fit(x, y)
            retrained = True
            steps_since_retrain = 0
        else:
            steps_since_retrain += 1

        rows.append(
            {
                "step": step,
                "angle": _angle(step, cfg),
                "true_auc": true_auc,
                "cbpe_estimate": band.estimate,
                "cbpe_lo": band.lo,
                "cbpe_hi": band.hi,
                "covariate_drift": bool(covariate),
                "retrained": retrained,
            }
        )
    return pd.DataFrame(rows)
