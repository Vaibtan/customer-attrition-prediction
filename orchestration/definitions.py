"""Dagster ``Definitions`` for the churn platform -- the code location Dagster loads.

Run the UI against this module::

    dagster dev -m orchestration.definitions

Phase 2 registers the vertical-slice assets + the ``SliceConfig`` resource. Later phases extend the
asset graph (partitioned backfills, drift-gated retrain) without changing this entrypoint.
"""

from __future__ import annotations

from dagster import Definitions

from orchestration.assets import (
    SliceConfig,
    offline_pit_features,
    online_features,
    parity_report,
    synthetic_dataset,
)

defs = Definitions(
    assets=[synthetic_dataset, offline_pit_features, online_features, parity_report],
    resources={"slice_config": SliceConfig()},
)
