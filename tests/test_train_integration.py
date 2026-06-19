"""End-to-end training on a sample — guards the whole leak-free pipeline wiring."""

from __future__ import annotations

from churn import config
from churn.pipeline import candidate_models
from churn.train import select_model, train_and_evaluate


def test_train_and_evaluate_sample(sample):
    art = train_and_evaluate(sample, persist=False)

    assert art["best_name"] in candidate_models()
    assert set(art["model_test_proba"]) == set(candidate_models())

    for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= art["metrics_business"][metric] <= 1.0

    assert 0.0 <= art["t_star"] <= 1.0
    # Tier cutpoints respect ordering (medium boundary never above high boundary).
    cut = art["tier_cutpoints"]
    assert cut["t_mid"] <= cut["t_star"]

    assert art["auc_ci"]["auc_lo"] <= art["auc_ci"]["auc_hi"]
    assert art["auc_diff_ci"]["diff_lo"] <= art["auc_diff_ci"]["diff_hi"]
    assert len(art["sensitivity"]) == len(config.COST_SCENARIOS)


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
