"""DomainRepair maps impossible values to NaN; Winsorizer learns caps on fit only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.cleaning import DomainRepair, Winsorizer
from tests.conftest import make_raw


def test_impossible_values_become_nan():
    df = pd.concat(
        [
            make_raw(account_age_days=-1),
            make_raw(days_since_last_login=999),
            make_raw(days_since_last_login=-5),
        ],
        ignore_index=True,
    )
    out = DomainRepair().fit_transform(df)
    assert np.isnan(out.loc[0, "account_age_days"])
    assert np.isnan(out.loc[1, "days_since_last_login"])
    assert np.isnan(out.loc[2, "days_since_last_login"])


def test_zero_orders_with_spend_is_preserved():
    df = make_raw(num_orders_last_90d=0, monthly_spend=80.0, avg_order_value=40.0)
    out = DomainRepair().fit_transform(df)
    assert out.loc[0, "monthly_spend"] == 80.0
    assert out.loc[0, "avg_order_value"] == 40.0
    assert "logically_inconsistent" not in out.columns


def test_winsorizer_caps_learned_on_fit_only():
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    w = Winsorizer(lower=0.0, upper=0.75).fit(train)
    cap = w.upper_[0]
    transformed = w.transform(np.array([[1000.0]]))
    assert transformed[0, 0] == cap
    cap2 = Winsorizer(lower=0.0, upper=0.75).fit(np.vstack([train, [[1e6]]])).upper_[0]
    assert cap2 > cap


def test_winsorizer_preserves_nan():
    arr = np.array([[1.0], [np.nan], [3.0]])
    out = Winsorizer().fit_transform(arr)
    assert np.isnan(out[1, 0])


def test_winsorizer_only_caps_selected_columns():
    df = pd.DataFrame({"spend": [1.0, 2.0, 3.0, 4.0, 1000.0], "flag": [0.0, 0.0, 0.0, 0.0, 1.0]})
    out = Winsorizer(columns=["spend"]).fit_transform(df)
    assert out["flag"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert out["spend"].max() < 1000.0
