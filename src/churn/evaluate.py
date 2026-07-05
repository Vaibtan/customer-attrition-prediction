"""Metrics, the cost-based decision threshold, and bootstrap/sensitivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype="float64")
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def expected_value(y_true, proba, threshold, cost, value, uplift) -> float:
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    return tp * uplift * value - (tp + fp) * cost


@dataclass(frozen=True)
class Economics:
    """Illustrative campaign economics (see WRITEUP Part 5); not measured from the data."""

    cost: float
    value: float
    uplift: float

    @property
    def break_even(self) -> float:
        """Probability above which contacting a customer is +EV in expectation."""
        return self.cost / (self.value * self.uplift)

    @classmethod
    def default(cls) -> Economics:
        return cls(config.COST_PER_CONTACT, config.VALUE_PER_RETAINED, config.CAMPAIGN_UPLIFT)

    @classmethod
    def from_mapping(cls, params) -> Economics:
        return cls(float(params["cost"]), float(params["value"]), float(params["uplift"]))


@dataclass(frozen=True)
class CampaignThreshold:
    """The cost-based decision boundary and its risk tiers, fit on one score distribution.

    Owns t_star (grid-optimal), the EV curve, break-even, and the medium/high tier
    cutpoints together, so callers ask one object for the boundary instead of threading
    t_star between a threshold search and a separate tier computation.
    """

    economics: Economics
    t_star: float
    best_ev: float
    grid: np.ndarray
    ev_curve: np.ndarray
    t_mid: float

    @classmethod
    def fit(cls, y_true, proba, economics: Economics, grid=None) -> CampaignThreshold:
        y_true = np.asarray(y_true)
        proba = np.asarray(proba, dtype="float64")
        if grid is None:
            grid = np.linspace(0.0, 1.0, 101)
        ev = np.array(
            [
                expected_value(y_true, proba, t, economics.cost, economics.value, economics.uplift)
                for t in grid
            ]
        )
        best = int(np.argmax(ev))
        t_star = float(grid[best])
        below = proba[proba < t_star]
        t_mid = float(np.median(below)) if below.size else float(t_star / 2.0)
        return cls(economics, t_star, float(ev[best]), grid, ev, t_mid)

    @property
    def break_even(self) -> float:
        return self.economics.break_even

    @property
    def cutpoints(self) -> dict:
        """Frozen tier boundaries persisted to the registry and shared by batch + API."""
        return {"t_mid": self.t_mid, "t_star": self.t_star}

    def ev(self, y_true, proba, threshold: float | None = None) -> float:
        """Campaign EV on (y, proba) at this object's economics; defaults to t_star."""
        t = self.t_star if threshold is None else threshold
        return expected_value(
            y_true, proba, t, self.economics.cost, self.economics.value, self.economics.uplift
        )


def _bootstrap_metrics(
    y_true: np.ndarray, metrics: dict, n_rounds: int, seed: int
) -> dict[str, np.ndarray]:
    """Resample row indices once; score each named metric on every valid resample.

    ``metrics`` maps a name to a callable ``idx -> float``. A single ``(n_rounds, n)`` index draw is
    shared across all metrics -- so paired diffs (and combined ROC/PR bands) score identical
    resamples -- and single-class resamples (``< 2`` classes present) are skipped. Returns
    ``{name: array of per-round scores}``. The draw sequence matches the old per-function bootstraps
    exactly (same ``default_rng(seed).integers(0, n, (n_rounds, n))``), so results are unchanged.
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_rounds, n))
    out: dict[str, list[float]] = {name: [] for name in metrics}
    for idx in draws:
        if np.unique(y_true[idx]).size < 2:
            continue
        for name, fn in metrics.items():
            out[name].append(fn(idx))
    return {name: np.asarray(vals) for name, vals in out.items()}


def _quantile_band(values: np.ndarray, alpha: float) -> tuple[float, float]:
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def _diff_band(diffs: np.ndarray, alpha: float) -> dict:
    lo, hi = _quantile_band(diffs, alpha)
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": lo,
        "diff_hi": hi,
        "n_rounds": int(diffs.size),
    }


def bootstrap_auc_ci(
    y_true,
    proba,
    n_rounds: int = config.BOOTSTRAP_ROUNDS,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for ROC-AUC; single-class resamples are skipped."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype="float64")
    aucs = _bootstrap_metrics(
        y_true, {"auc": lambda idx: roc_auc_score(y_true[idx], proba[idx])}, n_rounds, seed
    )["auc"]
    lo, hi = _quantile_band(aucs, alpha)
    return {"auc_lo": lo, "auc_hi": hi, "n_rounds": int(aucs.size)}


def bootstrap_auc_diff_ci(
    y_true,
    proba_a,
    proba_b,
    n_rounds: int = config.BOOTSTRAP_ROUNDS,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for AUC(a) - AUC(b); a CI straddling 0 means a tie."""
    y_true = np.asarray(y_true)
    a = np.asarray(proba_a, dtype="float64")
    b = np.asarray(proba_b, dtype="float64")
    diffs = _bootstrap_metrics(
        y_true,
        {"d": lambda idx: roc_auc_score(y_true[idx], a[idx]) - roc_auc_score(y_true[idx], b[idx])},
        n_rounds,
        seed,
    )["d"]
    return _diff_band(diffs, alpha)


def bootstrap_pr_auc_diff_ci(
    y_true,
    proba_a,
    proba_b,
    n_rounds: int = config.BOOTSTRAP_ROUNDS,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for PR-AUC(a) - PR-AUC(b); a CI straddling 0 means a tie.

    The PR-AUC counterpart to ``bootstrap_auc_diff_ci`` (average precision) -- required by the
    promotion gate (Sec 4.8), which demands BOTH lower bounds clear the MDE. Same paired resamples
    (each round scores both models on the identical indices) so the difference is properly paired.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(proba_a, dtype="float64")
    b = np.asarray(proba_b, dtype="float64")
    diffs = _bootstrap_metrics(
        y_true,
        {
            "d": lambda idx: (
                average_precision_score(y_true[idx], a[idx])
                - average_precision_score(y_true[idx], b[idx])
            )
        },
        n_rounds,
        seed,
    )["d"]
    return _diff_band(diffs, alpha)


def bootstrap_diff_ci(
    y_true,
    proba_a,
    proba_b,
    n_rounds: int = config.BOOTSTRAP_ROUNDS,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> dict:
    """Paired ΔROC-AUC and ΔPR-AUC CIs from ONE shared resample matrix.

    Equivalent to calling :func:`bootstrap_auc_diff_ci` and :func:`bootstrap_pr_auc_diff_ci` with
    the same ``seed`` (identical draws), but scores both metrics on the one draw instead of
    regenerating it -- the promotion gate needs both, so this halves the bootstrap work. Returns
    ``{"roc": <diff band>, "pr": <diff band>}``.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(proba_a, dtype="float64")
    b = np.asarray(proba_b, dtype="float64")
    scored = _bootstrap_metrics(
        y_true,
        {
            "roc": lambda idx: (
                roc_auc_score(y_true[idx], a[idx]) - roc_auc_score(y_true[idx], b[idx])
            ),
            "pr": lambda idx: (
                average_precision_score(y_true[idx], a[idx])
                - average_precision_score(y_true[idx], b[idx])
            ),
        },
        n_rounds,
        seed,
    )
    return {name: _diff_band(vals, alpha) for name, vals in scored.items()}


def threshold_sensitivity(y_val, val_proba, y_test, test_proba, scenarios) -> list[dict]:
    """Recompute t* (on validation) and report hold-out EV per economic scenario."""
    y_test = np.asarray(y_test)
    n_test = len(y_test)
    rows = []
    for name, params in scenarios.items():
        econ = Economics.from_mapping(params)
        threshold = CampaignThreshold.fit(y_val, val_proba, econ)
        ev_target = threshold.ev(y_test, test_proba)
        ev_all = threshold.ev(y_test, test_proba, threshold=0.0)
        rows.append(
            {
                "scenario": name,
                "cost": econ.cost,
                "value": econ.value,
                "uplift": econ.uplift,
                "break_even": econ.break_even,
                "t_star": threshold.t_star,
                "ev_target_per_1k": float(ev_target / n_test * 1000.0),
                "ev_all_per_1k": float(ev_all / n_test * 1000.0),
            }
        )
    return rows


def confusion_at(y_true, proba, threshold: float):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), pred)
