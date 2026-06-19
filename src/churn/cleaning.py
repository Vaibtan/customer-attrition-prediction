"""Cleaning transformers — sklearn-compatible so they live *inside* the pipeline.

``DomainRepair`` is stateless (it applies fixed domain rules), so it can never
leak. ``Winsorizer`` learns caps from data and therefore MUST be fit on the
training fold only — which is exactly why it is a transformer and not a
pre-split mutation of the dataframe (the starter's mistake).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class DomainRepair(BaseEstimator, TransformerMixin):
    """Map logically-impossible values to NaN and flag inconsistent records.

    Stateless: the valid ranges come from domain knowledge, not from the data,
    so applying this before the split introduces no leakage. We still keep it in
    the pipeline so batch/online scoring repairs inputs identically.
    """

    def __init__(self, valid_ranges: dict | None = None):
        self.valid_ranges = valid_ranges

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        ranges = self.valid_ranges or config.VALID_RANGES

        for col, (lo, hi) in ranges.items():
            if col not in X.columns:
                continue
            X[col] = X[col].astype("float64")
            if lo is not None:
                X.loc[X[col] < lo, col] = np.nan
            if hi is not None:
                X.loc[X[col] > hi, col] = np.nan

        # NB: we deliberately do NOT treat `num_orders_last_90d == 0` with positive
        # `monthly_spend` as inconsistent. Spend is a 6-month average while orders
        # are a 90-day count, so a customer who purchased in months 4-6 but not in
        # the last 90 days is valid (and exactly the lapsing signal we want to keep).
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip selected columns to learned [lower, upper] quantiles (caps, not deletes).

    Only the columns in ``columns`` are capped (default: every column); all others
    pass through untouched. This matters because winsorizing a rare binary flag
    would clip every positive to 0 (the p99 of a ~0.7%-prevalence flag is 0),
    silently deleting the feature.

    Rationale vs. the starter's IQR row-deletion: deleting rows by a global rule
    also drops test records and changes the evaluation population. Capping keeps
    every row's other signal while neutralising leverage points, and the caps are
    learned on the training fold only.
    """

    def __init__(
        self,
        columns: list | None = None,
        lower: float = config.WINSOR_LOWER,
        upper: float = config.WINSOR_UPPER,
    ):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype="float64")
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"x{i}" for i in range(arr.shape[1])]
        self.feature_names_in_ = np.asarray(names)

        if self.columns is None:
            self.winsor_idx_ = list(range(arr.shape[1]))
        else:
            idx = {n: i for i, n in enumerate(names)}
            self.winsor_idx_ = [idx[c] for c in self.columns if c in idx]

        sel = arr[:, self.winsor_idx_] if self.winsor_idx_ else arr[:, :0]
        self.lower_ = np.nanquantile(sel, self.lower, axis=0)
        self.upper_ = np.nanquantile(sel, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype="float64").copy()
        for j, col in enumerate(self.winsor_idx_):
            arr[:, col] = np.clip(arr[:, col], self.lower_[j], self.upper_[j])  # NaN passes
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, columns=X.columns, index=X.index)
        return arr

    def get_feature_names_out(self, input_features=None):
        # Always return a name array (never None) to honour the sklearn contract.
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(self.feature_names_in_)
