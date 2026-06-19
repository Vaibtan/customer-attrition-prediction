"""End-to-end training on a sample — guards the whole leak-free pipeline wiring."""

from __future__ import annotations

from churn import config
from churn.evaluate import CampaignThreshold
from churn.pipeline import candidate_models
from churn.train import HoldoutReport, TrainingResult, select_model, train_and_evaluate


def test_train_and_evaluate_sample(sample):
    result = train_and_evaluate(sample, persist=False)

    assert isinstance(result, TrainingResult)
    assert isinstance(result.threshold, CampaignThreshold)
    assert isinstance(result.holdout, HoldoutReport)

    assert result.best_name in candidate_models()
    h = result.holdout
    assert set(h.model_test_proba) == set(candidate_models())

    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= h.metrics_business[metric] <= 1.0

    assert 0.0 <= result.threshold.t_star <= 1.0
    # Tier cutpoints respect ordering (medium boundary never above high boundary).
    cut = result.threshold.cutpoints
    assert cut["t_mid"] <= cut["t_star"]

    assert h.auc_ci["auc_lo"] <= h.auc_ci["auc_hi"]
    assert h.auc_diff_ci["diff_lo"] <= h.auc_diff_ci["diff_hi"]
    assert h.runner_up in candidate_models() and h.runner_up != result.best_name
    assert len(h.sensitivity) == len(config.COST_SCENARIOS)


def test_select_model_breaks_ties_toward_simplest():
    tied = {
        "logistic_regression": {"cv_auc_mean": 0.592, "cv_auc_std": 0.02},
        "random_forest": {"cv_auc_mean": 0.594, "cv_auc_std": 0.02},
        "hist_gradient_boosting": {"cv_auc_mean": 0.565, "cv_auc_std": 0.02},
    }
    assert select_model(tied) == "logistic_regression"

    clear = {
        "logistic_regression": {"cv_auc_mean": 0.55, "cv_auc_std": 0.02},
        "random_forest": {"cv_auc_mean": 0.70, "cv_auc_std": 0.02},
        "hist_gradient_boosting": {"cv_auc_mean": 0.60, "cv_auc_std": 0.02},
    }
    assert select_model(clear) == "random_forest"
