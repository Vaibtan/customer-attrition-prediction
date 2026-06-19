"""Metric correctness — guards the starter's Bug #2 (AUC from hard labels)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from churn.evaluate import (
    bootstrap_auc_ci,
    bootstrap_auc_diff_ci,
    classification_metrics,
    confusion_at,
    expected_value,
    select_threshold,
    threshold_sensitivity,
    tier_cutpoints,
)


def test_roc_auc_uses_probabilities_not_labels():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y_true, proba, threshold=0.5)
    # Correct AUC (0.75) is computed from probabilities ...
    assert np.isclose(m["roc_auc"], roc_auc_score(y_true, proba))
    assert np.isclose(m["roc_auc"], 0.75)
    # ... and differs from the degenerate label-based value (0.5) the starter produced.
    hard = (proba >= 0.5).astype(int)
    assert not np.isclose(roc_auc_score(y_true, proba), roc_auc_score(y_true, hard))


def test_precision_recall_f1_known_case():
    y_true = np.array([0, 1, 1, 0])
    proba = np.array([0.2, 0.9, 0.4, 0.6])  # preds @0.5 = [0,1,0,1]
    m = classification_metrics(y_true, proba, threshold=0.5)
    assert np.isclose(m["precision"], 0.5)  # 1 TP / (1 TP + 1 FP)
    assert np.isclose(m["recall"], 0.5)  # 1 TP / (1 TP + 1 FN)
    assert np.isclose(m["f1"], 0.5)


def test_expected_value_and_threshold_selection():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.9])
    # Contacting the two true churners (cost 5, value 100, uplift 0.3) is positive;
    # contacting the negatives is not. Optimal threshold separates them.
    t_star, best_ev, grid, ev = select_threshold(y_true, proba, cost=5, value=100, uplift=0.3)
    assert 0.2 < t_star <= 0.7
    assert np.isclose(best_ev, expected_value(y_true, proba, t_star, 5, 100, 0.3))
    assert best_ev == ev.max()


def test_tier_cutpoints_ordering():
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    cut = tier_cutpoints(proba, t_star=0.5)
    assert cut["t_star"] == 0.5
    assert cut["t_mid"] <= cut["t_star"]


def test_confusion_at_uses_threshold():
    cm = confusion_at([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.6], threshold=0.5)
    assert cm.shape == (2, 2)
    # preds @0.5 = [0,1,0,1] → TN=1, FP=1, FN=1, TP=1
    assert cm.tolist() == [[1, 1], [1, 1]]


def test_bootstrap_auc_ci_bounds_and_order():
    # Perfectly separable → every valid resample scores AUC 1.0, so the CI is [1, 1].
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.9])
    ci = bootstrap_auc_ci(y_true, proba, n_rounds=200, seed=0)
    assert ci["auc_lo"] <= ci["auc_hi"]
    assert 0.0 <= ci["auc_lo"] <= 1.0 and 0.0 <= ci["auc_hi"] <= 1.0
    assert np.isclose(ci["auc_lo"], 1.0) and np.isclose(ci["auc_hi"], 1.0)


def test_bootstrap_auc_diff_ci_straddles_zero_for_identical_models():
    # Identical predictions → the AUC gap is exactly 0 every round, CI = [0, 0].
    y_true = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.2, 0.3, 0.6, 0.8, 0.1, 0.7])
    d = bootstrap_auc_diff_ci(y_true, proba, proba, n_rounds=200, seed=0)
    assert d["diff_lo"] <= 0.0 <= d["diff_hi"]
    assert np.isclose(d["diff_mean"], 0.0)


def test_threshold_sensitivity_break_even_and_shape():
    y_val = np.array([0, 0, 1, 1])
    val_proba = np.array([0.1, 0.2, 0.7, 0.9])
    y_test = np.array([0, 1, 1, 0])
    test_proba = np.array([0.2, 0.8, 0.6, 0.3])
    scenarios = {
        "a": {"cost": 5.0, "value": 100.0, "uplift": 0.5},  # break-even 0.1
        "b": {"cost": 20.0, "value": 200.0, "uplift": 0.2},  # break-even 0.5
    }
    rows = threshold_sensitivity(y_val, val_proba, y_test, test_proba, scenarios)
    assert [r["scenario"] for r in rows] == ["a", "b"]
    assert np.isclose(rows[0]["break_even"], 0.1)
    assert np.isclose(rows[1]["break_even"], 0.5)
    for r in rows:
        assert 0.0 <= r["t_star"] <= 1.0
        assert {"ev_target_per_1k", "ev_all_per_1k"} <= set(r)
