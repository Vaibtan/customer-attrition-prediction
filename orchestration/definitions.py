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

# The scoring timeline is a STATIC, historical partition set (a frozen world produces no new
# data), so materialising it is what a backfill IS -- launched from the UI/CLI, never a cron. The
# old weekly cron on this partitioned job was semantically incoherent AND broken: a bare
# ScheduleDefinition tick emits a RunRequest with no partition_key, which a partitioned job
# cannot launch (REVIEW_ISSUES.md REV-03).
timeline_backfill_job = define_asset_job(
    "timeline_backfill", selection=AssetSelection.assets(timeline_events, pit_snapshot)
)

# The recurring decision belongs to the UNPARTITIONED retrain branch -- "weekly drift-gated
# retrain" is the story the docs tell, and a bare ScheduleDefinition is correct here.
weekly_retrain_job = define_asset_job(
    "weekly_retrain", selection=AssetSelection.assets(drift_gated_retrain)
)

weekly_retrain_schedule = ScheduleDefinition(
    name="weekly_retrain",
    job=weekly_retrain_job,
    cron_schedule="0 6 * * 1",  # Mondays 06:00 -- detect drift, retrain + gate only if it fires
)

defs = Definitions(
    assets=_slice_assets + _timeline_assets,
    jobs=[timeline_backfill_job, weekly_retrain_job],
    schedules=[weekly_retrain_schedule],
    resources={"slice_config": SliceConfig(), "retrain_scenario": RetrainScenario()},
)
