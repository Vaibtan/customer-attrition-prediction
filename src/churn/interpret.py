"""Model interpretability: odds ratios, permutation importance, reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def clean_names(names) -> list[str]:
    return [str(n).removeprefix("num__").removeprefix("cat__") for n in names]


def feature_names(fitted_pipeline) -> list[str]:
    return clean_names(fitted_pipeline.named_steps["pre"].get_feature_names_out())


def linear_clf(fitted_pipeline):
    clf = fitted_pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError(
            f"Expected a linear model with coef_, got {type(clf).__name__}; "
            "odds ratios / reason codes require a linear estimator."
        )
    return clf


def odds_ratios(fitted_linear_pipeline) -> pd.DataFrame:
    names = feature_names(fitted_linear_pipeline)
    coef = np.ravel(linear_clf(fitted_linear_pipeline).coef_)
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    return df.reindex(df["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def permutation_importance_df(model, X, y, scoring="roc_auc", n_repeats=10, seed=42):
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed
    )
    df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def reason_codes_for_frame(base_linear_pipeline, X, k: int = 3) -> list[str]:
    """Top-k signed contributions (coef * standardized value) per row."""
    coef = np.ravel(linear_clf(base_linear_pipeline).coef_)
    pre = base_linear_pipeline[:-1]
    z = np.asarray(pre.transform(X), dtype="float64")
    names = feature_names(base_linear_pipeline)
    contrib = z * coef

    out = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:k]
        parts = [f"{names[j]}({'+' if row[j] >= 0 else '-'})" for j in idx]
        out.append("; ".join(parts))
    return out
