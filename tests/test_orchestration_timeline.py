"""Phase-5 orchestration depth: a partitioned backfill + a drift-gated conditional retrain.

Runs in-process via Dagster's ``materialize`` (no external infra): materialise multiple timeline
partitions (the backfill, distinct per-t0 outputs) and exercise BOTH sides of the conditional
retrain branch -- drift present -> retrain + gate runs; no drift -> retrain skipped.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dagster")

from dagster import materialize  # noqa: E402

from orchestration import definitions as D  # noqa: E402
from orchestration import timeline as T  # noqa: E402
from orchestration.assets import SliceConfig  # noqa: E402


def test_partitioned_backfill_materialises_distinct_snapshots():
    resources = {"slice_config": SliceConfig(n_customers=40)}
    means = {}
    for key in ("2024-12-11", "2025-01-01"):
        result = materialize(
            [T.timeline_events, T.pit_snapshot],
            partition_key=key,
            resources=resources,
        )
        assert result.success
        mat = result.asset_materializations_for_node("pit_snapshot")[0]
        meta = mat.metadata
        assert meta["t0"].value == key
        means[key] = meta["mean_login_count_90d"].value
    # An earlier scoring instant sees fewer recent logins than the anchor -> the backfill partitions
    # are genuinely different point-in-time snapshots, not copies.
    assert means["2024-12-11"] != means["2025-01-01"]


def test_drift_triggers_conditional_retrain():
    result = materialize(
        [T.drift_gated_retrain],
        resources={"retrain_scenario": T.RetrainScenario(inject_drift=True, n=1500)},
    )
    assert result.success
    meta = result.asset_materializations_for_node("drift_gated_retrain")[0].metadata
    assert meta["drift_detected"].value is True
    assert meta["retrained"].value is True


def test_no_drift_skips_retrain():
    result = materialize(
        [T.drift_gated_retrain],
        resources={"retrain_scenario": T.RetrainScenario(inject_drift=False, n=1500)},
    )
    assert result.success
    meta = result.asset_materializations_for_node("drift_gated_retrain")[0].metadata
    assert meta["drift_detected"].value is False
    assert meta["retrained"].value is False


def test_definitions_expose_backfill_job_and_schedule():
    assert D.timeline_backfill_job.name == "timeline_backfill"
    schedule_names = {s.name for s in D.defs.schedules}
    assert "weekly_timeline" in schedule_names
    keys = {k.to_user_string() for k in D.defs.resolve_all_asset_keys()}
    assert {"pit_snapshot", "drift_gated_retrain", "timeline_events"} <= keys
