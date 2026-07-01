"""Offline feature store (DIY, DuckDB + Parquet) -- point-in-time features via ASOF + windows.

The batch (production) scoring path: the event log -> **offline PIT features** keyed by
``(customer_id, t0)``. The temporal contract (ENHANCEMENT_PLAN.md §4.3) is enforced in SQL --
``event_ts <= t0`` (strict PIT; the label window ``(t0, t0+90d]`` never enters a feature) -- and
every feature is a deterministic aggregate of the event set **deduplicated by ``event_id``** so the
online Quix aggregation (Phase 2) can reproduce it byte-for-byte under late/duplicate/reordered
streams (D6.5 parity constraint). Recency uses an explicit **ASOF LEFT JOIN** (the reviewable PIT
centerpiece, §7); counts/means use windowed conditional aggregation. Empty windows resolve to the
frozen D6.6 sentinels so an absent key is byte-identical offline and online.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# Frozen-at-Stage-2 feature windows (days). Exploratory now; pinned in analysis_spec.json (D3/D6.5).
# Multi-scale login windows (14/28/90d) + the direct-readout means (session depth, sentiment) are
# the informative proxies of the recent latent health h(t0) validated in Phase-1 exploration (D7).
DEFAULT_FEATURE_SPEC: dict = {
    "recency_sentinel_days": 365.0,  # D6.6: no login in window -> full-window recency
    "windows": {
        "near_days": 14,
        "short_days": 28,
        "mid_days": 90,
        "pay_days": 180,
        "full_days": 365,
    },
}

# The offline feature columns (order fixed for reproducible frames + the analysis-code hash).
FEATURE_COLUMNS: list[str] = [
    "days_since_last_login",
    "login_count_14d",
    "login_count_28d",
    "login_count_90d",
    "login_trend_28_90",
    "mean_session_depth_28d",
    "mean_session_depth_90d",
    "has_login_90d",
    "order_count_90d",
    "support_count_90d",
    "mean_sentiment_90d",
    "has_support_90d",
    "payment_fail_count_180d",
    "downgrade_count_365d",
]

_EVENT_SCHEMA = ["event_id", "customer_id", "event_ts", "event_type", "value"]


def compute_pit_features(
    events: pd.DataFrame, cohort: pd.DataFrame, spec: dict = DEFAULT_FEATURE_SPEC
) -> pd.DataFrame:
    """PIT feature frame keyed by ``(customer_id, t0)`` for every cohort row.

    ``events`` is the long log (``event_id, customer_id, event_ts, event_type, value``); ``cohort``
    is ``(customer_id, t0)``. Computed in DuckDB: an ASOF LEFT JOIN for login recency + a windowed
    conditional aggregation for the counts/means, merged and sentinel-filled (D6.6).
    """
    win = spec["windows"]
    con = duckdb.connect()
    try:
        # De-duplicate by event_id up front: the offline aggregates are then idempotent to a
        # redelivered duplicate, matching the online store's exactly-once semantics (D6.5).
        ev = events[_EVENT_SCHEMA].drop_duplicates(subset="event_id")
        con.register("events_raw", ev)
        con.register("cohort", cohort[["customer_id", "t0"]])
        agg = con.execute(_AGG_SQL.format(**win)).df()
        recency = con.execute(_RECENCY_SQL).df()
    finally:
        con.close()

    feats = cohort[["customer_id", "t0"]].merge(agg, on=["customer_id", "t0"], how="left")
    feats = feats.merge(recency, on=["customer_id", "t0"], how="left")
    return _finalize(feats, spec)


def _finalize(feats: pd.DataFrame, spec: dict) -> pd.DataFrame:
    sentinel = spec["recency_sentinel_days"]
    # Recency: t0 - last_login_ts in days; the D6.6 sentinel when no prior login exists.
    delta = (feats["t0"] - feats["last_login_ts"]).dt.total_seconds() / 86_400.0
    feats["days_since_last_login"] = delta.fillna(sentinel).clip(upper=sentinel)
    # Counts default to 0 on an empty window; means default to the 0.0 sentinel (+ presence flags).
    for col in (
        "login_count_14d",
        "login_count_28d",
        "login_count_90d",
        "order_count_90d",
        "support_count_90d",
        "payment_fail_count_180d",
        "downgrade_count_365d",
    ):
        feats[col] = feats[col].fillna(0.0)
    feats["has_login_90d"] = (feats["login_count_90d"] > 0).astype(float)
    feats["has_support_90d"] = (feats["support_count_90d"] > 0).astype(float)
    feats["mean_session_depth_28d"] = feats["mean_session_depth_28d"].fillna(0.0)
    feats["mean_session_depth_90d"] = feats["mean_session_depth_90d"].fillna(0.0)
    feats["mean_sentiment_90d"] = feats["mean_sentiment_90d"].fillna(0.0)
    # Recent-fraction proxy of the login trend (bounded, reorder/duplicate-safe).
    feats["login_trend_28_90"] = feats["login_count_28d"] / (feats["login_count_90d"] + 1.0)
    return feats[["customer_id", "t0", *FEATURE_COLUMNS]]


# Windowed conditional aggregation. ``e.event_ts <= c.t0`` is the strict PIT filter; per-window
# COUNTs/AVGs use FILTER on the event type + a lookback interval measured from each row's own t0.
_AGG_SQL = """
SELECT
    c.customer_id,
    c.t0,
    COUNT(*) FILTER (
        WHERE e.event_type = 'login' AND e.event_ts > c.t0 - INTERVAL {near_days} DAY
    ) AS login_count_14d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'login' AND e.event_ts > c.t0 - INTERVAL {short_days} DAY
    ) AS login_count_28d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'login' AND e.event_ts > c.t0 - INTERVAL {mid_days} DAY
    ) AS login_count_90d,
    AVG(e.value) FILTER (
        WHERE e.event_type = 'login' AND e.event_ts > c.t0 - INTERVAL {short_days} DAY
    ) AS mean_session_depth_28d,
    AVG(e.value) FILTER (
        WHERE e.event_type = 'login' AND e.event_ts > c.t0 - INTERVAL {mid_days} DAY
    ) AS mean_session_depth_90d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'order' AND e.event_ts > c.t0 - INTERVAL {mid_days} DAY
    ) AS order_count_90d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'support' AND e.event_ts > c.t0 - INTERVAL {mid_days} DAY
    ) AS support_count_90d,
    AVG(e.value) FILTER (
        WHERE e.event_type = 'support' AND e.event_ts > c.t0 - INTERVAL {mid_days} DAY
    ) AS mean_sentiment_90d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'payment_fail' AND e.event_ts > c.t0 - INTERVAL {pay_days} DAY
    ) AS payment_fail_count_180d,
    COUNT(*) FILTER (
        WHERE e.event_type = 'downgrade' AND e.event_ts > c.t0 - INTERVAL {full_days} DAY
    ) AS downgrade_count_365d
FROM cohort c
LEFT JOIN events_raw e
    ON e.customer_id = c.customer_id AND e.event_ts <= c.t0
GROUP BY c.customer_id, c.t0
"""

# Login recency via an explicit ASOF LEFT JOIN: the nearest login at/before t0 (PIT-correct), NULL
# when a customer never logged in (-> the D6.6 sentinel in _finalize). ASOF LEFT keeps every row.
_RECENCY_SQL = """
SELECT c.customer_id, c.t0, l.event_ts AS last_login_ts
FROM cohort c
ASOF LEFT JOIN (SELECT customer_id, event_ts FROM events_raw WHERE event_type = 'login') l
    ON c.customer_id = l.customer_id AND c.t0 >= l.event_ts
"""


# --- Parquet durable store (D6.3) ------------------------------------------------------------

_TABLES = {"events": _EVENT_SCHEMA, "customers": None, "labels": None, "cohort": None}


def persist_dataset(frames: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    """Write each dataset frame to ``out_dir/<name>.parquet`` (LF-neutral columnar bytes)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def load_frame(name: str, store_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(store_dir) / f"{name}.parquet")
