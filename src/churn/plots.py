"""All figure generation. Uses the non-interactive Agg backend (no plt.show),
so the entry point runs unattended in CI / from a clean checkout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    auc,
    roc_curve,
)

from . import config  # noqa: E402

_PRETTY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def eda_overview(df, name="eda_overview.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    df[config.TARGET].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=["#4c72b0", "#dd8452"]
    )
    axes[0].set_title(f"Target balance (churn rate = {df[config.TARGET].mean():.1%})")
    axes[0].set_xticklabels(["Retained", "Churned"], rotation=0)

    by_plan = df.groupby("subscription_plan")[config.TARGET].mean().sort_values()
    by_plan.plot(kind="bar", ax=axes[1], color="#55a868")
    axes[1].set_title("Churn rate by subscription plan")
    axes[1].set_ylabel("Churn rate")
    axes[1].tick_params(axis="x", rotation=30)

    by_region = df.groupby("region")[config.TARGET].mean().sort_values()
    by_region.plot(kind="bar", ax=axes[2], color="#c44e52")
    axes[2].set_title("Churn rate by region")
    axes[2].set_ylabel("Churn rate")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("Exploratory overview", fontsize=13)
    return _save(fig, name)


def behavioural_by_churn(df, name="behavioural_by_churn.png"):
    cols = ["days_since_last_login", "monthly_spend", "support_tickets_raised"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cols, strict=True):
        data = [
            df.loc[df[config.TARGET] == 0, col].dropna(),
            df.loc[df[config.TARGET] == 1, col].dropna(),
        ]
        # Clip spend for legibility (outliers handled in modelling, not here).
        if col == "monthly_spend":
            cap = df[col].quantile(0.95)
            data = [d.clip(upper=cap) for d in data]
        ax.boxplot(data, tick_labels=["Retained", "Churned"], showfliers=False)
        ax.set_title(col)
    fig.suptitle("Behavioural features by churn outcome", fontsize=13)
    return _save(fig, name)


def roc_curves(model_test_proba, y_test, name="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, proba in model_test_proba.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{_PRETTY.get(model_name, model_name)} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    return _save(fig, name)


def pr_curve(y_test, proba, name="pr_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
    ax.axhline(np.mean(y_test), ls="--", color="grey", label="Base rate")
    ax.set_title("Precision–Recall curve (best model)")
    ax.legend(loc="upper right")
    return _save(fig, name)


def confusion(cm, threshold, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Not churned", "Churned"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, name)


def calibration(y_test, proba, name="calibration_curve.png"):
    from sklearn.calibration import CalibrationDisplay

    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=ax)
    ax.set_title("Reliability diagram (calibrated best model)")
    return _save(fig, name)


def permutation_importance(imp_df, name="feature_importance.png"):
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#8172b3")
    ax.set_title("Permutation importance (ROC-AUC drop)")
    ax.set_xlabel("Mean importance")
    return _save(fig, name)


def expected_value(grid, ev_curve, t_star, name="expected_value.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, ev_curve, color="#4c72b0")
    ax.axvline(t_star, ls="--", color="#dd8452", label=f"t* = {t_star:.2f}")
    ax.axhline(0, ls=":", color="grey")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected campaign value (USD)")
    ax.set_title("Expected value vs. threshold")
    ax.legend()
    return _save(fig, name)
