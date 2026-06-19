"""Model interpretability — sklearn-native (no SHAP).

For the linear winner this is both cheaper and more honest than SHAP:
- global: standardized-coefficient odds ratios + permutation importance
- local: per-customer reason codes from signed linear contributions (coef · z)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _clean_names(names) -> list[str]:
    """Strip ColumnTransformer prefixes (``num__``/``cat__``) for readability."""
    return [str(n).removeprefix("num__").removeprefix("cat__") for n in names]


def feature_names(fitted_pipeline) -> list[str]:
    return _clean_names(fitted_pipeline.named_steps["pre"].get_feature_names_out())


def _linear_clf(fitted_pipeline):
    """Return the final estimator, requiring it to expose linear ``coef_``.

    Odds ratios and reason codes are only meaningful for a linear model. Fail
    loudly here rather than with a cryptic ``AttributeError`` if a tree model is
    ever passed in (only logistic regression takes this path today).
    """
    clf = fitted_pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError(
            f"Expected a linear model with coef_, got {type(clf).__name__}; "
            "odds ratios / reason codes require a linear estimator."
        )
    return clf


def odds_ratios(fitted_linear_pipeline) -> pd.DataFrame:
    """Odds ratios for a fitted Logistic Regression pipeline.

    Features are standardized, so coefficients are directly comparable. OR > 1
    raises churn odds; OR < 1 lowers them (per 1 SD of the feature).
    """
    names = feature_names(fitted_linear_pipeline)
    coef = np.ravel(_linear_clf(fitted_linear_pipeline).coef_)
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    return df.reindex(df["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def permutation_importance_df(model, X, y, scoring="roc_auc", n_repeats=10, seed=42):
    """Unbiased importance (unlike RF impurity importance, the starter's choice)."""
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
    """Top-k signed contributions (coef · standardized value) per row.

    Uses the *uncalibrated* linear base pipeline (calibration is monotone, so the
    ranking of drivers is preserved). Returns human-readable strings like
    ``days_since_last_login(+); subscription_plan_Free(+); recency_ratio(-)``.
    """
    coef = np.ravel(_linear_clf(base_linear_pipeline).coef_)
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
