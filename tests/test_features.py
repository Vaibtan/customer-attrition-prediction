"""FeatureEngineer adds the expected columns without introducing div-by-zero."""

from __future__ import annotations

import numpy as np

from churn.features import FeatureEngineer
from tests.conftest import make_raw


def test_adds_expected_columns():
    out = FeatureEngineer().fit_transform(make_raw())
    for col in ("recency_ratio", "is_dormant", "support_per_order"):
        assert col in out.columns


def test_recency_ratio_safe_when_age_zero():
    out = FeatureEngineer().fit_transform(make_raw(account_age_days=0, days_since_last_login=10))
    assert np.isfinite(out.loc[0, "recency_ratio"])
    assert out.loc[0, "recency_ratio"] == 10 / 1


def test_is_dormant_threshold():
    hot = FeatureEngineer().fit_transform(make_raw(days_since_last_login=10))
    cold = FeatureEngineer().fit_transform(make_raw(days_since_last_login=200))
    assert hot.loc[0, "is_dormant"] == 0.0
    assert cold.loc[0, "is_dormant"] == 1.0


def test_support_per_order_no_div_zero():
    out = FeatureEngineer().fit_transform(make_raw(num_orders_last_90d=0, support_tickets_raised=2))
    assert np.isfinite(out.loc[0, "support_per_order"])
    assert out.loc[0, "support_per_order"] == 2 / 1
