"""Stateless, hypothesis-driven feature engineering; kept only if it earns lift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived churn-hypothesis features; NaNs propagate to downstream imputation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        age = X["account_age_days"]
        login = X["days_since_last_login"]

        X["recency_ratio"] = login / (age + 1)

        dormant = (login > config.DORMANT_DAYS).astype("float64")
        dormant[login.isna()] = np.nan
        X["is_dormant"] = dormant

        X["support_per_order"] = X["support_tickets_raised"] / (
            X["num_orders_last_90d"].fillna(0) + 1
        )
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.RAW_FEATURE_COLUMNS
        added = config.FEATURE_ENGINEER_OUTPUTS
        return np.asarray(base + [c for c in added if c not in base])
