"""Customer Churn Prediction — end-to-end entry point.
Runs the full exercise from a clean checkout::
    uv run python churn_prediction.py

Produces every figure in ``reports/figures/``, prints the EDA summary, the model
bake-off, and the hold-out metrics, and persists the chosen model to the
``models/`` registry. All preprocessing is leak-free (fit inside CV folds); the
reported ROC-AUC comes from probabilities, not hard labels.

This thin script orchestrates the ``churn`` package; the logic and the fixes for
the starter's bugs live there (see WRITEUP.md for the full defect catalog).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without an install step, in case `uv sync` has not
# installed it editable yet.
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from churn import config, evaluate, interpret, plots  # noqa: E402
from churn.data import load_data  # noqa: E402
from churn.train import train_and_evaluate  # noqa: E402


def _print_eda(df: pd.DataFrame) -> None:
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


def _print_metrics(art: dict) -> None:
    print("\n" + "=" * 70)
    print("MODEL BAKE-OFF (RepeatedStratifiedKFold ROC-AUC on train)")
    for name, r in sorted(art["cv_results"].items(), key=lambda kv: -kv[1]["cv_auc_mean"]):
        print(f"  {name:24s} AUC = {r['cv_auc_mean']:.3f} +/- {r['cv_auc_std']:.3f}")
    print(
        f"\nSelected model: {art['best_name']}  "
        f"(simplest within {config.MODEL_SELECTION_TOLERANCE:.2f} AUC of best -- a tie-break)"
    )
    d = art["auc_diff_ci"]
    tie = "straddles 0 => statistical tie" if d["diff_lo"] <= 0 <= d["diff_hi"] else "separated"
    print(
        f"  Hold-out AUC vs runner-up ({art['runner_up']}): "
        f"diff {d['diff_mean']:+.3f}  95% CI [{d['diff_lo']:+.3f}, {d['diff_hi']:+.3f}]  -> {tie}"
    )

    print("\nPER-MODEL TEST METRICS (uncalibrated, threshold = 0.50)")
    print(f"{'model':<24}{'precision':>10}{'recall':>9}{'f1':>8}{'roc_auc':>9}")
    for name, m in sorted(art["per_model_metrics"].items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(
            f"{name:<24}{m['precision']:>10.3f}{m['recall']:>9.3f}"
            f"{m['f1']:>8.3f}{m['roc_auc']:>9.3f}"
        )

    cut = art["tier_cutpoints"]
    print(
        f"\nCost-based threshold t* = {art['t_star']:.3f}  "
        f"(tier cutpoints: medium >= {cut['t_mid']:.3f}, high >= {cut['t_star']:.3f})"
    )

    print("\nHOLD-OUT METRICS")
    print(f"{'metric':<12}{'@ t* (business)':>18}{'@ 0.50':>12}")
    b, h = art["metrics_business"], art["metrics_half"]
    for m in ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]:
        print(f"{m:<12}{b[m]:>18.3f}{h[m]:>12.3f}")
    ci = art["auc_ci"]
    print(f"(winner hold-out ROC-AUC 95% bootstrap CI: [{ci['auc_lo']:.3f}, {ci['auc_hi']:.3f}])")

    print("\nEV SENSITIVITY TO CAMPAIGN ECONOMICS (illustrative -- $ are conditional)")
    print(
        f"{'scenario':<16}{'cost':>6}{'value':>7}{'uplift':>8}"
        f"{'break_even':>12}{'t*':>7}{'$/1k target':>13}{'$/1k all':>11}"
    )
    for s in art["sensitivity"]:
        print(
            f"{s['scenario']:<16}{s['cost']:>6.0f}{s['value']:>7.0f}{s['uplift']:>8.2f}"
            f"{s['break_even']:>12.2f}{s['t_star']:>7.2f}"
            f"{s['ev_target_per_1k']:>13.0f}{s['ev_all_per_1k']:>11.0f}"
        )


def main() -> None:
    df = load_data()
    _print_eda(df)

    print("\nGenerating EDA figures...")
    plots.eda_overview(df)
    plots.behavioural_by_churn(df)

    print("Training (bake-off -> calibrate -> cost-based threshold)...")
    art = train_and_evaluate(df)
    _print_metrics(art)

    print("\nGenerating evaluation figures...")
    plots.roc_curves(art["model_test_proba"], art["y_test"])
    plots.pr_curve(art["y_test"], art["test_proba"])
    plots.expected_value(art["ev_grid"], art["ev_curve"], art["t_star"])
    cm = evaluate.confusion_at(art["y_test"], art["test_proba"], art["t_star"])
    plots.confusion(cm, art["t_star"])
    plots.calibration(art["y_test"], art["test_proba"])

    imp = interpret.permutation_importance_df(
        art["calibrated"], art["X_test"], art["y_test"], seed=config.SEED
    )
    plots.permutation_importance(imp)

    if art["base_linear"] is not None:
        print("\nTop churn drivers (odds ratios, standardized):")
        print(interpret.odds_ratios(art["base_linear"]).head(8).to_string(index=False))

    print(f"\nFigures saved to: {config.FIGURES_DIR}")
    print(f"Model registered at: {art.get('run_dir')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
