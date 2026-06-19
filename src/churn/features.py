"""Feature engineering — stateless, hypothesis-driven transformers.

Each feature encodes a churn hypothesis (relative inactivity, dormancy, support
friction). They are stateless, so they're safe anywhere in the pipeline; whether
they actually *help* is measured in training and reported honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features. NaNs propagate and are imputed downstream."""

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        age = X["account_age_days"]
        login = X["days_since_last_login"]

        # Inactivity relative to tenure: a 60-day gap means more for a 90-day-old
        # account than a 3-year-old one. (+1 guards division by zero.)
        X["recency_ratio"] = login / (age + 1)

        # Step-change dormancy flag; preserve NaN where recency is unknown.
        dormant = (login > config.DORMANT_DAYS).astype("float64")
        dormant[login.isna()] = np.nan
        X["is_dormant"] = dormant

        # Support friction per unit of activity.
        X["support_per_order"] = X["support_tickets_raised"] / (
            X["num_orders_last_90d"].fillna(0) + 1
        )

        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.RAW_FEATURE_COLUMNS
        added = config.FEATURE_ENGINEER_OUTPUTS
        return np.asarray(base + [c for c in added if c not in base])
