"""Cleaning transformers that live inside the pipeline so they fit per CV fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class DomainRepair(BaseEstimator, TransformerMixin):
    """Map out-of-range values to NaN using fixed domain rules (stateless)."""

    def __init__(self, valid_ranges: dict | None = None):
        self.valid_ranges = valid_ranges

    def fit(self, X, y=None):
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
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip the named columns to learned [lower, upper] quantiles; others pass through."""

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
        names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(arr.shape[1])]
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
            arr[:, col] = np.clip(arr[:, col], self.lower_[j], self.upper_[j])
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, columns=X.columns, index=X.index)
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(self.feature_names_in_)
