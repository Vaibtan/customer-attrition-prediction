"""The analysis model + the leak-free, group-aware measurement contract (Stage-2 hashed).

The instrument is a tie-aware logistic regression (inheriting the base project's model family) over
one of two frozen feature sets: ``static`` (the 6 truly-static customer attributes) or
``static_event`` (static + the PIT event features). Out-of-fold predictions come from a
**StratifiedGroupKFold by ``customer_id``** so no customer straddles the split (§4.7) and the
preprocessor is refit inside each fold (no preprocessing leakage). ``customer_id`` is never a
feature (no ID memorisation). This module is part of the ANALYSIS set hashed at Stage 2 (D1/D2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config
from churn.featurestore import offline as OFF
from churn.simulator import params as P

# The two frozen feature sets (D5.3 static/event partition).
STATIC_CATEGORICAL = ["region", "device_type", "subscription_plan"]
STATIC_NUMERIC = list(P.STATIC_NUMERICS)  # account_age_days, monthly_spend, avg_order_value
STATIC_FEATURES = STATIC_CATEGORICAL + STATIC_NUMERIC
EVENT_FEATURES = list(OFF.FEATURE_COLUMNS)  # all numeric

FEATURE_SETS = {"static": STATIC_FEATURES, "static_event": STATIC_FEATURES + EVENT_FEATURES}
CV_SPLITS = 5


@dataclass(frozen=True)
class Design:
    """The assembled design matrix + label + customer groups for one experiment."""

    X: pd.DataFrame
    y: NDArray[np.int_]
    groups: NDArray


def assemble_design(dataset, pit_features: pd.DataFrame, use_real_label: bool = False) -> Design:
    """Join static attributes to PIT event features on ``customer_id``; pick the label column."""
    label_col = "real_churned" if use_real_label else "y"
    static = dataset.customers[["customer_id", *STATIC_FEATURES]]
    feats = static.merge(pit_features, on="customer_id", how="inner")
    labels = dataset.labels[["customer_id", label_col]]
    frame = feats.merge(labels, on="customer_id", how="inner")
    y = frame[label_col].to_numpy(dtype=int)
    groups = frame["customer_id"].to_numpy()
    x = frame[STATIC_FEATURES + EVENT_FEATURES]
    return Design(X=x, y=y, groups=groups)


def build_pipeline(feature_set: str, seed: int = config.SEED) -> Pipeline:
    """Preprocessor (impute + scale numerics, one-hot categoricals) + tie-aware LogReg."""
    columns = FEATURE_SETS[feature_set]
    numeric = [c for c in columns if c not in STATIC_CATEGORICAL]
    categorical = [c for c in columns if c in STATIC_CATEGORICAL]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=seed))])


def stratified_group_folds(y: NDArray, groups: NDArray, seed: int, n_splits: int = CV_SPLITS):
    """StratifiedGroupKFold splits by ``customer_id`` (leak-free, class-balanced, group-aware)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def group_oof_proba(
    feature_set: str, X: pd.DataFrame, y: NDArray, groups: NDArray, seed: int = config.SEED
) -> NDArray[np.float64]:
    """Leak-free out-of-fold P(churn): preprocessing + model refit inside each group-aware fold."""
    pipe = build_pipeline(feature_set, seed=seed)
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        pipe, X[FEATURE_SETS[feature_set]], y, cv=cv, groups=groups, method="predict_proba"
    )
    return proba[:, 1]


def auc(y: NDArray, proba: NDArray) -> float:
    return float(roc_auc_score(y, proba))


def pr_auc(y: NDArray, proba: NDArray) -> float:
    return float(average_precision_score(y, proba))
