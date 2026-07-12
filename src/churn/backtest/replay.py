"""Backtest replay harness -- a drifting timeline with an HONEST label horizon (REV-02).

Labels for the window scored at step ``t`` are only revealed at ``t + label_horizon_steps``:
``realized_auc`` at step ``t`` is the ground truth of step ``t - h``, and a retrain can only ever
fit the latest LABELED window. The horizon is **stylized** (default 3 steps; the real 90-day churn
label at weekly steps would be h~13, which makes any 14-step demo degenerate) and said so.

Two trigger policies over the same world (:func:`compare_triggers`):

- ``lagged_label`` -- retrain when the *revealed* (h-steps-old) AUC has decayed. Reacts to any
  drift kind, but always h steps late.
- ``label_free`` -- retrain when the covariate detector fires (no labels needed). Reacts
  immediately -- but ONLY to drift that moves the marginals.

Two drift kinds make both sides of that trade visible:

- ``concept`` (``max_shift=0``) -- the boundary rotates, marginals never move: the detector and
  CBPE are structurally blind (the module's original honesty story), only the lagged monitor
  recovers, h steps late.
- ``combined`` (``max_shift>0``) -- realistic drift is rarely pure: the marginals shift while the
  boundary rotates. The detector fires at drift start, so the label-free trigger recovers
  ~h steps before the lagged one -- "label-free monitoring buys you the horizon".

This is a synthetic-domain systems demonstration of the monitoring + lifecycle machinery, not a
real-world performance claim.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from churn.drift import cbpe
from churn.drift import detectors as DET

_FEATURES = ["f0", "f1"]
TRIGGERS = ("lagged_label", "label_free")


@dataclass(frozen=True)
class BacktestConfig:
    n_steps: int = 14
    drift_start: int = 4
    ramp_steps: int = 4  # steps over which the drift ramps to its max
    max_angle: float = 1.25  # radians the decision boundary rotates (concept component)
    max_shift: float = 0.0  # marginal mean-shift on f0 (covariate component; 0 = pure concept)
    window_n: int = 2500
    retrain_threshold: float = 0.72  # REVEALED (lagged) AUC below this triggers a retrain
    label_horizon_steps: int = 3  # stylized label delay -- see the module docstring
    trigger: str = "lagged_label"  # "lagged_label" | "label_free"
    seed: int = 0


def _progress(step: int, cfg: BacktestConfig) -> float:
    if step < cfg.drift_start:
        return 0.0
    return min(1.0, (step - cfg.drift_start) / max(cfg.ramp_steps, 1))


def _window(
    rng: np.random.Generator, n: int, angle: float, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 1.0, (n, 2))
    x[:, 0] += shift  # covariate component: the f0 marginal moves (detector-visible)
    boundary = np.cos(angle) * x[:, 0] + np.sin(angle) * x[:, 1]
    y = (boundary + rng.normal(0.0, 0.5, n) > 0).astype(int)
    return x, y


def _fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(x, y)


def run_backtest(cfg: BacktestConfig | None = None) -> pd.DataFrame:
    """Replay the drifting timeline; one row per step of estimated vs true vs REVEALED perf.

    Columns: ``true_auc`` is ground truth at ``t`` (chart-only -- the monitor cannot see it);
    ``realized_auc`` is what the monitor actually has at ``t`` (the truth about ``t - h``, NaN
    while nothing is labeled yet); ``covariate_drift`` is the label-free signal; ``retrained``
    marks steps where ``cfg.trigger`` fired and the champion was refit on the latest LABELED
    window.
    """
    cfg = cfg or BacktestConfig()
    if cfg.trigger not in TRIGGERS:
        raise ValueError(f"unknown trigger {cfg.trigger!r}; expected one of {TRIGGERS}")
    h = int(cfg.label_horizon_steps)
    rng = np.random.default_rng(cfg.seed)
    x_ref, y_ref = _window(rng, cfg.window_n, angle=0.0, shift=0.0)
    reference = pd.DataFrame(x_ref, columns=_FEATURES)
    champion = _fit(x_ref, y_ref)

    windows: list[tuple[np.ndarray, np.ndarray]] = []  # (x, y) per step; y revealed at t + h
    true_history: list[float] = []
    rows = []
    steps_since_retrain = 99
    for step in range(cfg.n_steps):
        progress = _progress(step, cfg)
        angle = cfg.max_angle * progress
        shift = cfg.max_shift * progress
        x, y = _window(rng, cfg.window_n, angle, shift)
        windows.append((x, y))

        proba = champion.predict_proba(x)[:, 1]
        true_auc = float(roc_auc_score(y, proba))  # ground truth -- chart-only at step t
        true_history.append(true_auc)
        realized_auc = true_history[step - h] if step >= h else float("nan")
        band = cbpe.estimate(proba, n_rounds=100, seed=cfg.seed + step)
        covariate = DET.detect_drift(
            reference, pd.DataFrame(x, columns=_FEATURES), _FEATURES, []
        ).covariate_shift

        if cfg.trigger == "lagged_label":
            should_retrain = not np.isnan(realized_auc) and realized_auc < cfg.retrain_threshold
        else:  # label_free: the detector needs no labels -- fires the moment marginals move
            should_retrain = bool(covariate)

        retrained = False
        if should_retrain and steps_since_retrain >= 2 and step >= h:
            latest_labeled = windows[step - h]  # a retrain can only fit REVEALED labels
            champion = _fit(*latest_labeled)
            retrained = True
            steps_since_retrain = 0
        else:
            steps_since_retrain += 1

        rows.append(
            {
                "step": step,
                "angle": angle,
                "shift": shift,
                "true_auc": true_auc,
                "realized_auc": realized_auc,
                "cbpe_estimate": band.estimate,
                "cbpe_lo": band.lo,
                "cbpe_hi": band.hi,
                "covariate_drift": bool(covariate),
                "retrained": retrained,
                "trigger": cfg.trigger,
            }
        )
    return pd.DataFrame(rows)


def compare_triggers(cfg: BacktestConfig | None = None) -> dict[str, pd.DataFrame]:
    """Both trigger policies over the IDENTICAL world (same seed -> same windows).

    The world generation consumes the rng identically in both runs (retraining draws nothing),
    so the two frames differ only in when recovery happens -- the comparison is causal, not
    seed noise.
    """
    cfg = cfg or BacktestConfig(max_shift=1.5)
    return {
        trigger: run_backtest(dataclasses.replace(cfg, trigger=trigger)) for trigger in TRIGGERS
    }
