"""Data loading and a fail-fast schema contract; no cleaning happens here."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected it under data/customer_data.csv "
            "relative to the repo root."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
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
    ids = df[config.ID_COL].reset_index(drop=True)
    y = df[config.TARGET].astype(int).reset_index(drop=True)
    X = df.drop(columns=[config.ID_COL, config.TARGET]).reset_index(drop=True)
    return X, y, ids
