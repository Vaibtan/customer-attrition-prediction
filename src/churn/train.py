"""Training orchestration: bake-off -> calibrate -> threshold -> persist."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from . import config, evaluate, registry
from .data import load_data, split_features_target, validate_schema
from .pipeline import build_pipeline, candidate_models


@dataclass(frozen=True)
class HoldoutReport:
    """How the candidates do on held-out data, plus the honesty statistics.

    Bundles the per-model evaluation sweep, the two bootstrap CIs, and the EV
    sensitivity table -- the four concepts that used to share the artifacts dict.
    """

    X_test: pd.DataFrame
    y_test: pd.Series
    test_proba: np.ndarray
    model_test_proba: dict[str, np.ndarray]
    metrics_business: dict
    metrics_half: dict
    per_model_metrics: dict
    runner_up: str
    auc_ci: dict
    auc_diff_ci: dict
    sensitivity: list[dict]


@dataclass(frozen=True)
class TrainingResult:
    """The training contract as one typed object; consumers read named attributes."""

    best_name: str
    cv_results: dict
    calibrated: object
    base_linear: object | None
    threshold: evaluate.CampaignThreshold
    holdout: HoldoutReport
    seed: int
    run_dir: str | None = None


def bakeoff(X_train, y_train, seed):
    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=seed
    )
    results = {}
    for name, model in candidate_models(seed).items():
        scores = cross_val_score(
            build_pipeline(model), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {"cv_auc_mean": float(scores.mean()), "cv_auc_std": float(scores.std())}
    return results


def select_model(cv_results: dict) -> str:
    """Pick the simplest model within MODEL_SELECTION_TOLERANCE of the best CV AUC."""
    best_auc = max(r["cv_auc_mean"] for r in cv_results.values())
    tied = {
        n
        for n, r in cv_results.items()
        if best_auc - r["cv_auc_mean"] <= config.MODEL_SELECTION_TOLERANCE
    }
    for name in config.MODEL_PREFERENCE_ORDER:
        if name in tied:
            return name
    return max(cv_results, key=lambda n: cv_results[n]["cv_auc_mean"])


def evaluate_holdout(
    calibrated,
    best_name: str,
    cv_results: dict,
    threshold: evaluate.CampaignThreshold,
    X_train,
    y_train,
    oof,
    X_test,
    y_test,
    seed: int,
) -> HoldoutReport:
    """Score the calibrated winner + every candidate on the hold-out and run the honesty stats."""
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    metrics_business = evaluate.classification_metrics(y_test, test_proba, threshold.t_star)
    metrics_half = evaluate.classification_metrics(y_test, test_proba, 0.5)

    model_test_proba = {}
    per_model_metrics = {}
    for name, model in candidate_models(seed).items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        model_test_proba[name] = proba
        per_model_metrics[name] = evaluate.classification_metrics(y_test, proba, 0.5)

    runner_up = max(
        (n for n in cv_results if n != best_name),
        key=lambda n: cv_results[n]["cv_auc_mean"],
    )
    auc_ci = evaluate.bootstrap_auc_ci(y_test, model_test_proba[best_name], seed=seed)
    auc_diff_ci = evaluate.bootstrap_auc_diff_ci(
        y_test, model_test_proba[best_name], model_test_proba[runner_up], seed=seed
    )
    sensitivity = evaluate.threshold_sensitivity(
        y_train, oof, y_test, test_proba, config.COST_SCENARIOS
    )

    return HoldoutReport(
        X_test=X_test,
        y_test=y_test,
        test_proba=test_proba,
        model_test_proba=model_test_proba,
        metrics_business=metrics_business,
        metrics_half=metrics_half,
        per_model_metrics=per_model_metrics,
        runner_up=runner_up,
        auc_ci=auc_ci,
        auc_diff_ci=auc_diff_ci,
        sensitivity=sensitivity,
    )


def training_metadata(result: TrainingResult) -> dict:
    """The registry metadata schema, derived from the typed result."""
    econ = result.threshold.economics
    h = result.holdout
    return {
        "model_name": result.best_name,
        "cv_results": result.cv_results,
        "selection": {
            "tolerance": config.MODEL_SELECTION_TOLERANCE,
            "runner_up": h.runner_up,
            "winner_auc_ci": h.auc_ci,
            "winner_vs_runner_up_auc_diff_ci": h.auc_diff_ci,
        },
        "threshold": result.threshold.t_star,
        "tier_cutpoints": result.threshold.cutpoints,
        "cost_params": {
            "cost_per_contact": econ.cost,
            "value_per_retained": econ.value,
            "campaign_uplift": econ.uplift,
        },
        "ev_sensitivity": h.sensitivity,
        "per_model_metrics_test_half_threshold": h.per_model_metrics,
        "metrics_test_business_threshold": h.metrics_business,
        "metrics_test_half_threshold": h.metrics_half,
        "expected_value_at_threshold": result.threshold.best_ev,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "seed": result.seed,
    }


def train_and_evaluate(df=None, seed: int = config.SEED, persist: bool = True) -> TrainingResult:
    if df is None:
        df = load_data()
    validate_schema(df)

    X, y, ids = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=seed, stratify=y
    )

    cv_results = bakeoff(X_train, y_train, seed)
    best_name = select_model(cv_results)
    best_model = candidate_models(seed)[best_name]

    base_best = build_pipeline(best_model)
    calibrated = CalibratedClassifierCV(base_best, method="sigmoid", cv=config.CV_FOLDS)

    inner = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(
        calibrated, X_train, y_train, cv=inner, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = evaluate.CampaignThreshold.fit(y_train, oof, evaluate.Economics.default())

    calibrated.fit(X_train, y_train)
    base_linear = None
    if best_name == "logistic_regression":
        base_linear = build_pipeline(candidate_models(seed)["logistic_regression"])
        base_linear.fit(X_train, y_train)

    holdout = evaluate_holdout(
        calibrated, best_name, cv_results, threshold, X_train, y_train, oof, X_test, y_test, seed
    )

    result = TrainingResult(
        best_name=best_name,
        cv_results=cv_results,
        calibrated=calibrated,
        base_linear=base_linear,
        threshold=threshold,
        holdout=holdout,
        seed=seed,
    )

    if persist:
        run_dir = registry.save_run(calibrated, training_metadata(result), base_model=base_linear)
        result = replace(result, run_dir=str(run_dir))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument("--smoke", action="store_true", help="Fast run on a sample.")
    args = parser.parse_args()

    data = load_data()
    if args.smoke:
        data = data.sample(n=min(400, len(data)), random_state=config.SEED)
    result = train_and_evaluate(data, persist=not args.smoke)
    print(f"Best model: {result.best_name}")
    print(f"Test ROC-AUC: {result.holdout.metrics_business['roc_auc']:.3f}")
    print(
        f"t* = {result.threshold.t_star:.3f}  "
        f"(train OOF expected value, total = {result.threshold.best_ev:.1f})"
    )
