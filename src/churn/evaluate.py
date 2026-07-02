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
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y_true), size=(n_rounds, len(y_true)))
    aucs = np.asarray(
        [
            roc_auc_score(y_true[idx], proba[idx])
            for idx in draws
            if np.unique(y_true[idx]).size >= 2
        ]
    )
    return {
        "auc_lo": float(np.quantile(aucs, alpha / 2)),
        "auc_hi": float(np.quantile(aucs, 1 - alpha / 2)),
        "n_rounds": int(aucs.size),
    }


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
    proba_a = np.asarray(proba_a, dtype="float64")
    proba_b = np.asarray(proba_b, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y_true), size=(n_rounds, len(y_true)))
    diffs = np.asarray(
        [
            roc_auc_score(y_true[idx], proba_a[idx]) - roc_auc_score(y_true[idx], proba_b[idx])
            for idx in draws
            if np.unique(y_true[idx]).size >= 2
        ]
    )
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": float(np.quantile(diffs, alpha / 2)),
        "diff_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "n_rounds": int(diffs.size),
    }


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
    proba_a = np.asarray(proba_a, dtype="float64")
    proba_b = np.asarray(proba_b, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y_true), size=(n_rounds, len(y_true)))
    diffs = np.asarray(
        [
            average_precision_score(y_true[idx], proba_a[idx])
            - average_precision_score(y_true[idx], proba_b[idx])
            for idx in draws
            if np.unique(y_true[idx]).size >= 2
        ]
    )
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": float(np.quantile(diffs, alpha / 2)),
        "diff_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "n_rounds": int(diffs.size),
    }


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
