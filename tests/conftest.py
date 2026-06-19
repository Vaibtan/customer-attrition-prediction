"""Shared fixtures and builders for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import config
from churn.data import load_data

DEFAULTS = {
    "region": "North",
    "device_type": "Mobile",
    "subscription_plan": "Free",
    "account_age_days": 500,
    "monthly_spend": 50.0,
    "num_orders_last_90d": 5,
    "avg_order_value": 100.0,
    "support_tickets_raised": 1,
    "days_since_last_login": 100,
    "pages_per_session": 8.0,
}


def make_raw(**overrides) -> pd.DataFrame:
    row = {**DEFAULTS, **overrides}
    return pd.DataFrame([row])[config.RAW_FEATURE_COLUMNS]


@pytest.fixture(scope="session")
def raw_full() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="session")
def sample(raw_full) -> pd.DataFrame:
    return raw_full.sample(n=300, random_state=config.SEED).reset_index(drop=True)
