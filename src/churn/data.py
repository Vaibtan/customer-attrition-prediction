"""Data loading and a fail-fast schema contract.

Loading does *no* cleaning — every fitted transformation lives inside the
modelling pipeline so it can be fit per CV fold (this is the structural fix for
the starter's data leakage). The only thing that happens here is reading the CSV
and validating that it matches the contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw customer CSV. Raises a clear error if the file is missing."""
    path = Path(path)  # idempotent if already a Path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected it under data/customer_data.csv "
            "relative to the repo root."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Fail-fast data contract. Raises ``ValueError`` on any violation.

    Training data must include the binary target. Scoring batches are allowed to
    be targetless, but still need the customer id and all raw feature columns.
    """
    expected = (
        config.EXPECTED_COLUMNS if require_target else [config.ID_COL, *config.RAW_FEATURE_COLUMNS]
    )
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate customer_id values found.")

    if require_target or config.TARGET in df.columns:
        bad = set(pd.unique(df[config.TARGET].dropna())) - {0, 1}
        if bad:
            raise ValueError(f"Target '{config.TARGET}' must be binary; saw {bad}.")


def split_features_target(df: pd.DataFrame):
    """Return (X, y, ids). X excludes id and target so the pipeline never sees them."""
    ids = df[config.ID_COL].reset_index(drop=True)
    y = df[config.TARGET].astype(int).reset_index(drop=True)
    X = df.drop(columns=[config.ID_COL, config.TARGET]).reset_index(drop=True)
    return X, y, ids
