"""Metrics, the cost-based decision threshold, and bootstrap/sensitivity helpers."""

from __future__ import annotations

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


def select_threshold(
    y_true,
    proba,
    cost: float = config.COST_PER_CONTACT,
    value: float = config.VALUE_PER_RETAINED,
    uplift: float = config.CAMPAIGN_UPLIFT,
    grid=None,
):
    """Return (t_star, best_ev, grid, ev_curve) maximising expected campaign value."""
    y_true = np.asarray(y_true)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    ev = np.array([expected_value(y_true, proba, t, cost, value, uplift) for t in grid])
    best = int(np.argmax(ev))
    return float(grid[best]), float(ev[best]), grid, ev


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


def threshold_sensitivity(y_val, val_proba, y_test, test_proba, scenarios) -> list[dict]:
    """Recompute t* (on validation) and report hold-out EV per economic scenario."""
    y_test = np.asarray(y_test)
    n_test = len(y_test)
    rows = []
    for name, p in scenarios.items():
        cost, value, uplift = p["cost"], p["value"], p["uplift"]
        t_star = select_threshold(y_val, val_proba, cost, value, uplift)[0]
        ev_target = expected_value(y_test, test_proba, t_star, cost, value, uplift)
        ev_all = expected_value(y_test, test_proba, 0.0, cost, value, uplift)
        rows.append(
            {
                "scenario": name,
                "cost": float(cost),
                "value": float(value),
                "uplift": float(uplift),
                "break_even": float(cost / (value * uplift)),
                "t_star": float(t_star),
                "ev_target_per_1k": float(ev_target / n_test * 1000.0),
                "ev_all_per_1k": float(ev_all / n_test * 1000.0),
            }
        )
    return rows


def tier_cutpoints(proba, t_star: float):
    """Medium/high tier boundaries, frozen on the validation score distribution."""
    proba = np.asarray(proba, dtype="float64")
    below = proba[proba < t_star]
    t_mid = float(np.median(below)) if below.size else float(t_star / 2.0)
    return {"t_mid": t_mid, "t_star": float(t_star)}


def confusion_at(y_true, proba, threshold: float):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), pred)
