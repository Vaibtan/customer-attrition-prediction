"""Metric correctness, threshold selection, and the bootstrap/sensitivity helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from churn.evaluate import (
    CampaignThreshold,
    Economics,
    bootstrap_auc_ci,
    bootstrap_auc_diff_ci,
    classification_metrics,
    confusion_at,
    expected_value,
    threshold_sensitivity,
)


def test_roc_auc_uses_probabilities_not_labels():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y_true, proba, threshold=0.5)
    assert np.isclose(m["roc_auc"], roc_auc_score(y_true, proba))
    assert np.isclose(m["roc_auc"], 0.75)
    hard = (proba >= 0.5).astype(int)
    assert not np.isclose(roc_auc_score(y_true, proba), roc_auc_score(y_true, hard))


def test_precision_recall_f1_known_case():
    y_true = np.array([0, 1, 1, 0])
    proba = np.array([0.2, 0.9, 0.4, 0.6])
    m = classification_metrics(y_true, proba, threshold=0.5)
    assert np.isclose(m["precision"], 0.5)
    assert np.isclose(m["recall"], 0.5)
    assert np.isclose(m["f1"], 0.5)


def test_campaign_threshold_selects_ev_maximising_boundary():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.9])
    ct = CampaignThreshold.fit(y_true, proba, Economics(cost=5, value=100, uplift=0.3))
    assert 0.2 < ct.t_star <= 0.7
    assert np.isclose(ct.best_ev, expected_value(y_true, proba, ct.t_star, 5, 100, 0.3))
    assert ct.best_ev == ct.ev_curve.max()
    assert np.isclose(ct.ev(y_true, proba), ct.best_ev)
    assert np.isclose(ct.break_even, 5 / (100 * 0.3))


def test_campaign_threshold_tiers_match_old_cutpoint_logic():
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    ct = CampaignThreshold.fit(y_true, proba, Economics(cost=20, value=200, uplift=0.2))
    below = proba[proba < ct.t_star]
    expected_mid = float(np.median(below)) if below.size else ct.t_star / 2.0
    assert ct.t_mid == expected_mid
    assert ct.cutpoints == {"t_mid": ct.t_mid, "t_star": ct.t_star}
    assert ct.t_mid <= ct.t_star


def test_confusion_at_uses_threshold():
    cm = confusion_at([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.6], threshold=0.5)
    assert cm.shape == (2, 2)
    assert cm.tolist() == [[1, 1], [1, 1]]


def test_bootstrap_auc_ci_bounds_and_order():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.7, 0.9])
    ci = bootstrap_auc_ci(y_true, proba, n_rounds=200, seed=0)
    assert ci["auc_lo"] <= ci["auc_hi"]
    assert 0.0 <= ci["auc_lo"] <= 1.0 and 0.0 <= ci["auc_hi"] <= 1.0
    assert np.isclose(ci["auc_lo"], 1.0) and np.isclose(ci["auc_hi"], 1.0)


def test_bootstrap_auc_diff_ci_straddles_zero_for_identical_models():
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
        "a": {"cost": 5.0, "value": 100.0, "uplift": 0.5},
        "b": {"cost": 20.0, "value": 200.0, "uplift": 0.2},
    }
    rows = threshold_sensitivity(y_val, val_proba, y_test, test_proba, scenarios)
    assert [r["scenario"] for r in rows] == ["a", "b"]
    assert np.isclose(rows[0]["break_even"], 0.1)
    assert np.isclose(rows[1]["break_even"], 0.5)
    for r in rows:
        assert 0.0 <= r["t_star"] <= 1.0
        assert {"ev_target_per_1k", "ev_all_per_1k"} <= set(r)
