"""Offline PIT feature tests (ENHANCEMENT_PLAN.md §4.3 / D6.5 / D6.6).

The offline store computes point-in-time features by DuckDB ASOF (recency) + windowed aggregation
over the event log. These tests pin the load-bearing contract: strict PIT (``event_ts <= t0``, the
label window untouched), the empty-window sentinels (D6.6), idempotency under duplicate delivery
(dedup by ``event_id`` -- the property the Phase-2 parity test relies on), and one feature row per
cohort key.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.featurestore import offline as OFF

T0 = pd.Timestamp("2025-01-01T00:00:00")


def _events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    if "value" not in df:
        df["value"] = pd.NA
    return df[["event_id", "customer_id", "event_ts", "event_type", "value"]]


def _cohort(ids: list[str], t0: pd.Timestamp = T0) -> pd.DataFrame:
    return pd.DataFrame({"customer_id": ids, "t0": [t0] * len(ids)})


def test_feature_frame_is_keyed_one_row_per_cohort_key():
    ev = _events(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=3),
                "event_type": "login",
                "value": 7.0,
            },
        ]
    )
    feats = OFF.compute_pit_features(ev, _cohort(["A", "B"]))
    assert list(feats.columns[:2]) == ["customer_id", "t0"]
    assert len(feats) == 2
    assert set(feats["customer_id"]) == {"A", "B"}
    assert set(OFF.FEATURE_COLUMNS) <= set(feats.columns)


def test_strict_pit_excludes_events_after_t0_includes_at_t0():
    ev = _events(
        [
            # exactly at t0 -> included (max(feature_ts) <= t0).
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0,
                "event_type": "login",
                "value": 5.0,
            },
            # one nanosecond after t0 -> excluded (belongs to the label window).
            {
                "event_id": "A:login:1",
                "customer_id": "A",
                "event_ts": T0 + pd.Timedelta(days=1),
                "event_type": "login",
                "value": 9.0,
            },
        ]
    )
    feats = OFF.compute_pit_features(ev, _cohort(["A"])).set_index("customer_id")
    assert feats.loc["A", "login_count_90d"] == 1.0  # only the t0 event counts
    assert feats.loc["A", "days_since_last_login"] == 0.0  # last login is exactly at t0


def test_recency_and_empty_window_sentinels():
    ev = _events(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=10),
                "event_type": "login",
                "value": 6.0,
            },
        ]
    )
    feats = OFF.compute_pit_features(ev, _cohort(["A", "Z"])).set_index("customer_id")
    # A logged in 10 days before t0.
    assert feats.loc["A", "days_since_last_login"] == pytest.approx(10.0)
    assert feats.loc["A", "has_login_90d"] == 1.0
    # Z has no events at all -> the D6.6 empty-window contract.
    assert feats.loc["Z", "days_since_last_login"] == 365.0
    assert feats.loc["Z", "login_count_90d"] == 0.0
    assert feats.loc["Z", "has_login_90d"] == 0.0
    assert feats.loc["Z", "mean_sentiment_90d"] == 0.0
    assert feats.loc["Z", "mean_session_depth_90d"] == 0.0


def test_windowed_counts_and_means_are_hand_correct():
    ev = _events(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=5),
                "event_type": "login",
                "value": 10.0,
            },
            {
                "event_id": "A:login:1",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=40),
                "event_type": "login",
                "value": 4.0,
            },
            {
                "event_id": "A:login:2",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=100),
                "event_type": "login",
                "value": 2.0,
            },  # outside 90d
            {
                "event_id": "A:sup:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=8),
                "event_type": "support",
                "value": -0.5,
            },
            {
                "event_id": "A:pay:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=100),
                "event_type": "payment_fail",
                "value": None,
            },  # within the 180d window
        ]
    )
    feats = OFF.compute_pit_features(ev, _cohort(["A"])).set_index("customer_id")
    assert feats.loc["A", "login_count_28d"] == 1.0  # only day-5
    assert feats.loc["A", "login_count_90d"] == 2.0  # day-5 + day-40
    assert feats.loc["A", "mean_session_depth_90d"] == pytest.approx(7.0)  # (10+4)/2
    assert feats.loc["A", "support_count_90d"] == 1.0
    assert feats.loc["A", "mean_sentiment_90d"] == pytest.approx(-0.5)
    assert feats.loc["A", "payment_fail_count_180d"] == 1.0  # 180-day window count


def test_idempotent_under_duplicate_event_id():
    # Dedup by event_id: a redelivered duplicate must not double-count (Phase-2 parity property).
    base = {
        "customer_id": "A",
        "event_ts": T0 - pd.Timedelta(days=3),
        "event_type": "login",
        "value": 5.0,
    }
    ev = _events([{"event_id": "A:login:0", **base}, {"event_id": "A:login:0", **base}])
    feats = OFF.compute_pit_features(ev, _cohort(["A"])).set_index("customer_id")
    assert feats.loc["A", "login_count_90d"] == 1.0  # duplicate collapsed


def test_no_feature_is_null():
    ev = _events(
        [
            {
                "event_id": "A:order:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=2),
                "event_type": "order",
                "value": None,
            },
        ]
    )
    feats = OFF.compute_pit_features(ev, _cohort(["A", "B", "C"]))
    assert feats[OFF.FEATURE_COLUMNS].notna().all().all()
