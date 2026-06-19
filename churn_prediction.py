"""End-to-end entry point: EDA, model bake-off, metrics, figures, registered model."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from churn import config, evaluate, interpret, plots  # noqa: E402
from churn.data import load_data  # noqa: E402
from churn.train import train_and_evaluate  # noqa: E402


def print_eda(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATASET SUMMARY")
    print(f"Shape: {df.shape}  (rows x columns)")
    print(f"Overall churn rate: {df[config.TARGET].mean():.4f}  (~ balanced)")

    print("\nColumn dtypes:")
    print(df.dtypes.to_string())

    miss = df.isnull().sum()
    nonzero_miss = miss[miss > 0]
    print("\nMissing values (non-zero):")
    print(nonzero_miss.to_string() if not nonzero_miss.empty else "  none")

    print("\nDescriptive statistics (numeric columns):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.describe().T.round(2).to_string())

    print("\nChurn rate by subscription_plan:")
    print(df.groupby("subscription_plan")[config.TARGET].mean().round(3).to_string())
    print("\nChurn rate by region:")
    print(df.groupby("region")[config.TARGET].mean().round(3).to_string())


def print_metrics(result) -> None:
    h = result.holdout
    print("\n" + "=" * 70)
    print("MODEL BAKE-OFF (RepeatedStratifiedKFold ROC-AUC on train)")
    for name, r in sorted(result.cv_results.items(), key=lambda kv: -kv[1]["cv_auc_mean"]):
        print(f"  {name:24s} AUC = {r['cv_auc_mean']:.3f} +/- {r['cv_auc_std']:.3f}")
    print(
        f"\nSelected model: {result.best_name}  "
        f"(simplest within {config.MODEL_SELECTION_TOLERANCE:.2f} AUC of best -- a tie-break)"
    )
    d = h.auc_diff_ci
    tie = "straddles 0 => statistical tie" if d["diff_lo"] <= 0 <= d["diff_hi"] else "separated"
    print(
        f"  Hold-out AUC vs runner-up ({h.runner_up}): "
        f"diff {d['diff_mean']:+.3f}  95% CI [{d['diff_lo']:+.3f}, {d['diff_hi']:+.3f}]  -> {tie}"
    )

    print("\nPER-MODEL TEST METRICS (uncalibrated, threshold = 0.50)")
    print(f"{'model':<24}{'precision':>10}{'recall':>9}{'f1':>8}{'roc_auc':>9}")
    for name, m in sorted(h.per_model_metrics.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(
            f"{name:<24}{m['precision']:>10.3f}{m['recall']:>9.3f}"
            f"{m['f1']:>8.3f}{m['roc_auc']:>9.3f}"
        )

    cut = result.threshold.cutpoints
    print(
        f"\nCost-based threshold t* = {result.threshold.t_star:.3f}  "
        f"(tier cutpoints: medium >= {cut['t_mid']:.3f}, high >= {cut['t_star']:.3f})"
    )

    print("\nHOLD-OUT METRICS")
    print(f"{'metric':<12}{'@ t* (business)':>18}{'@ 0.50':>12}")
    b, half = h.metrics_business, h.metrics_half
    for m in ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]:
        print(f"{m:<12}{b[m]:>18.3f}{half[m]:>12.3f}")
    ci = h.auc_ci
    print(f"(winner hold-out ROC-AUC 95% bootstrap CI: [{ci['auc_lo']:.3f}, {ci['auc_hi']:.3f}])")

    print("\nEV SENSITIVITY TO CAMPAIGN ECONOMICS (illustrative -- $ are conditional)")
    print(
        f"{'scenario':<16}{'cost':>6}{'value':>7}{'uplift':>8}"
        f"{'break_even':>12}{'t*':>7}{'$/1k target':>13}{'$/1k all':>11}"
    )
    for s in h.sensitivity:
        print(
            f"{s['scenario']:<16}{s['cost']:>6.0f}{s['value']:>7.0f}{s['uplift']:>8.2f}"
            f"{s['break_even']:>12.2f}{s['t_star']:>7.2f}"
            f"{s['ev_target_per_1k']:>13.0f}{s['ev_all_per_1k']:>11.0f}"
        )


def main() -> None:
    df = load_data()
    print_eda(df)

    print("\nGenerating EDA figures...")
    plots.eda_overview(df)
    plots.behavioural_by_churn(df)

    print("Training (bake-off -> calibrate -> cost-based threshold)...")
    result = train_and_evaluate(df)
    print_metrics(result)

    h, threshold = result.holdout, result.threshold
    print("\nGenerating evaluation figures...")
    plots.roc_curves(h.model_test_proba, h.y_test)
    plots.pr_curve(h.y_test, h.test_proba)
    plots.expected_value(threshold.grid, threshold.ev_curve, threshold.t_star)
    cm = evaluate.confusion_at(h.y_test, h.test_proba, threshold.t_star)
    plots.confusion(cm, threshold.t_star)
    plots.calibration(h.y_test, h.test_proba)

    imp = interpret.permutation_importance_df(
        result.calibrated, h.X_test, h.y_test, seed=config.SEED
    )
    plots.permutation_importance(imp)

    if result.base_linear is not None:
        print("\nTop churn drivers (odds ratios, standardized):")
        print(interpret.odds_ratios(result.base_linear).head(8).to_string(index=False))

    print(f"\nFigures saved to: {config.FIGURES_DIR}")
    print(f"Model registered at: {result.run_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
