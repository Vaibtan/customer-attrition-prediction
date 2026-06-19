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


def signed_coefficients(fitted_linear_pipeline) -> pd.Series:
    """Coefficients indexed by feature name; the one place name<->coef alignment is asserted."""
    names = feature_names(fitted_linear_pipeline)
    coef = np.ravel(linear_clf(fitted_linear_pipeline).coef_)
    if len(names) != coef.size:
        raise ValueError(
            f"name<->coef misalignment: {len(names)} feature names vs {coef.size} coefficients."
        )
    return pd.Series(coef, index=names, name="coef")


def contributions(fitted_linear_pipeline, X) -> pd.DataFrame:
    """Signed per-row contributions (coef * standardized value); columns ARE the feature names.

    Returning a name-labelled frame makes the name<->coef<->z pairing structural: a caller
    cannot read a contribution without the feature it belongs to.
    """
    coef = signed_coefficients(fitted_linear_pipeline)
    z = np.asarray(fitted_linear_pipeline[:-1].transform(X), dtype="float64")
    if z.shape[1] != coef.size:
        raise ValueError(
            f"z<->coef misalignment: transformed matrix has {z.shape[1]} columns "
            f"vs {coef.size} coefficients."
        )
    index = X.index if hasattr(X, "index") else None
    return pd.DataFrame(z * coef.to_numpy(), columns=list(coef.index), index=index)


def odds_ratios(fitted_linear_pipeline) -> pd.DataFrame:
    coef = signed_coefficients(fitted_linear_pipeline)
    df = pd.DataFrame(
        {"feature": coef.index, "coef": coef.to_numpy(), "odds_ratio": np.exp(coef.to_numpy())}
    )
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
    contrib = contributions(base_linear_pipeline, X)
    names = contrib.columns.to_numpy()

    out = []
    for row in contrib.to_numpy():
        idx = np.argsort(-np.abs(row))[:k]
        parts = [f"{names[j]}({'+' if row[j] >= 0 else '-'})" for j in idx]
        out.append("; ".join(parts))
    return out
