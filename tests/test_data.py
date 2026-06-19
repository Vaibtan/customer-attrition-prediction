"""Data loading + the fail-fast schema contract."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import config
from churn.data import split_features_target, validate_schema


def test_real_data_passes_contract(raw_full):
    validate_schema(raw_full)
    assert list(raw_full.columns) == config.EXPECTED_COLUMNS


def test_missing_column_rejected(raw_full):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(raw_full.drop(columns=["monthly_spend"]))


def test_targetless_scoring_batch_passes_contract(raw_full):
    scoring_batch = raw_full.drop(columns=[config.TARGET])
    validate_schema(scoring_batch, require_target=False)


def test_targetless_scoring_batch_still_requires_features(raw_full):
    scoring_batch = raw_full.drop(columns=[config.TARGET, "monthly_spend"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(scoring_batch, require_target=False)


def test_non_binary_target_rejected(raw_full):
    bad = raw_full.copy()
    bad.loc[bad.index[0], config.TARGET] = 7
    with pytest.raises(ValueError, match="must be binary"):
        validate_schema(bad)


def test_split_excludes_id_and_target(raw_full):
    X, y, ids = split_features_target(raw_full)
    assert config.ID_COL not in X.columns
    assert config.TARGET not in X.columns
    assert len(X) == len(y) == len(ids)
    assert set(pd.unique(y)) <= {0, 1}
