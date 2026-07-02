"""Dagster ``Definitions`` for the churn platform -- the code location Dagster loads.

Run the UI against this module::

    dagster dev -m orchestration.definitions

Phase 2 registered the vertical-slice assets + ``SliceConfig``. Phase 5 adds the partitioned
timeline (``pit_snapshot``) + its backfill job/schedule and the drift-gated conditional-retrain
branch (``drift_gated_retrain`` + ``RetrainScenario``).
"""

from __future__ import annotations

from dagster import AssetSelection, Definitions, ScheduleDefinition, define_asset_job

from orchestration.assets import (
    SliceConfig,
    offline_pit_features,
    online_features,
    parity_report,
    synthetic_dataset,
)
from orchestration.timeline import (
    RetrainScenario,
    drift_gated_retrain,
    pit_snapshot,
    timeline_events,
)

_slice_assets = [synthetic_dataset, offline_pit_features, online_features, parity_report]
_timeline_assets = [timeline_events, pit_snapshot, drift_gated_retrain]

timeline_backfill_job = define_asset_job(
    "timeline_backfill", selection=AssetSelection.assets(timeline_events, pit_snapshot)
)

weekly_timeline_schedule = ScheduleDefinition(
    name="weekly_timeline",
    job=timeline_backfill_job,
    cron_schedule="0 6 * * 1",  # Mondays 06:00 -- score the latest window
)

defs = Definitions(
    assets=_slice_assets + _timeline_assets,
    jobs=[timeline_backfill_job],
    schedules=[weekly_timeline_schedule],
    resources={"slice_config": SliceConfig(), "retrain_scenario": RetrainScenario()},
)
