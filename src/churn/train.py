"""Training orchestration: bake-off -> calibrate -> threshold -> persist."""

from __future__ import annotations

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


def train_and_evaluate(df=None, seed: int = config.SEED, persist: bool = True) -> dict:
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
    t_star, best_ev, ev_grid, ev_curve = evaluate.select_threshold(y_train, oof)
    cutpoints = evaluate.tier_cutpoints(oof, t_star)

    calibrated.fit(X_train, y_train)
    base_linear = None
    if best_name == "logistic_regression":
        base_linear = build_pipeline(candidate_models(seed)["logistic_regression"])
        base_linear.fit(X_train, y_train)

    test_proba = calibrated.predict_proba(X_test)[:, 1]
    metrics_business = evaluate.classification_metrics(y_test, test_proba, t_star)
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

    artifacts = {
        "best_name": best_name,
        "cv_results": cv_results,
        "calibrated": calibrated,
        "base_linear": base_linear,
        "t_star": t_star,
        "best_ev": best_ev,
        "ev_grid": ev_grid,
        "ev_curve": ev_curve,
        "tier_cutpoints": cutpoints,
        "X_test": X_test,
        "y_test": y_test,
        "test_proba": test_proba,
        "model_test_proba": model_test_proba,
        "per_model_metrics": per_model_metrics,
        "runner_up": runner_up,
        "auc_ci": auc_ci,
        "auc_diff_ci": auc_diff_ci,
        "sensitivity": sensitivity,
        "metrics_business": metrics_business,
        "metrics_half": metrics_half,
        "cost_params": {
            "cost_per_contact": config.COST_PER_CONTACT,
            "value_per_retained": config.VALUE_PER_RETAINED,
            "campaign_uplift": config.CAMPAIGN_UPLIFT,
        },
    }

    if persist:
        metadata = {
            "model_name": best_name,
            "cv_results": cv_results,
            "selection": {
                "tolerance": config.MODEL_SELECTION_TOLERANCE,
                "runner_up": runner_up,
                "winner_auc_ci": auc_ci,
                "winner_vs_runner_up_auc_diff_ci": auc_diff_ci,
            },
            "threshold": t_star,
            "tier_cutpoints": cutpoints,
            "cost_params": artifacts["cost_params"],
            "ev_sensitivity": sensitivity,
            "per_model_metrics_test_half_threshold": per_model_metrics,
            "metrics_test_business_threshold": metrics_business,
            "metrics_test_half_threshold": metrics_half,
            "expected_value_at_threshold": best_ev,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "seed": seed,
        }
        artifacts["run_dir"] = str(registry.save_run(calibrated, metadata, base_model=base_linear))

    return artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument("--smoke", action="store_true", help="Fast run on a sample.")
    args = parser.parse_args()

    data = load_data()
    if args.smoke:
        data = data.sample(n=min(400, len(data)), random_state=config.SEED)
    art = train_and_evaluate(data, persist=not args.smoke)
    print(f"Best model: {art['best_name']}")
    print(f"Test ROC-AUC: {art['metrics_business']['roc_auc']:.3f}")
    print(f"t* = {art['t_star']:.3f}  (train OOF expected value, total = {art['best_ev']:.1f})")
