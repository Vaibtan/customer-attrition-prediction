"""Phase-5 orchestration depth: a partitioned timeline backfill + a drift-gated retrain branch.

- ``pit_snapshot`` is **partitioned** over a set of scoring instants (``t0``): each partition
  computes the PIT feature snapshot as of its own ``t0`` from the shared event log, so a backfill
  materialises one run per partition (temporal lineage).
- ``drift_gated_retrain`` is the **conditional** branch: it runs the type-aware detectors and only
  retrains + gates when drift fires (reusing the Phase-4 units), with a ``RetryPolicy`` for
  transient failures. A ``RetrainScenario`` resource drives whether drift is present, so the branch
  is demonstrable both ways.

(No ``from __future__ import annotations``: Dagster introspects real context annotations.)
"""

import numpy as np
import pandas as pd
from dagster import (
    AssetExecutionContext,
    ConfigurableResource,
    MaterializeResult,
    RetryPolicy,
    StaticPartitionsDefinition,
    asset,
)

from churn.drift import detectors as DET
from churn.drift import simulate as SIM
from churn.lifecycle.adapters import ColumnSubsetModel
from churn.lifecycle.retrain import retrain_and_gate
from churn.lifecycle.synthetic import linear_boundary_xy
from orchestration.assets import SliceConfig

# Weekly scoring instants leading up to the anchor t0 -- the timeline to back-fill over.
TIMELINE = StaticPartitionsDefinition(["2024-12-11", "2024-12-18", "2024-12-25", "2025-01-01"])
_SYNTH_FEATURES = ["f0", "f1", "f2"]


@asset
def timeline_events(slice_config: SliceConfig) -> dict:
    """Generate the event log + cohort ONCE; the partitioned snapshots share it."""
    from churn.simulator import generate as G
    from churn.simulator import params as P

    ds = G.build_population_dataset(P.load_params(), seed=slice_config.seed, n_synthetic=0)
    ids = ds.customers["customer_id"].tolist()[: slice_config.n_customers]
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)
    return {"events": events, "customer_ids": ids}


@asset(partitions_def=TIMELINE)
def pit_snapshot(
    context: AssetExecutionContext, timeline_events: dict, slice_config: SliceConfig
) -> MaterializeResult:
    """PIT features as of THIS partition's t0 -- one run per partition in a backfill."""
    from churn.featurestore import offline as OFF
    from churn.featurestore.cohort import make_cohort

    t0 = pd.Timestamp(context.partition_key)
    ids = timeline_events["customer_ids"]
    cohort = make_cohort(ids, t0)
    pit = OFF.compute_pit_features(timeline_events["events"], cohort)
    return MaterializeResult(
        metadata={
            "t0": context.partition_key,
            "rows": len(pit),
            "mean_login_count_90d": float(pit["login_count_90d"].mean()),
        }
    )


class RetrainScenario(ConfigurableResource):
    """Drives the conditional branch: whether the current window carries drift."""

    inject_drift: bool = True
    n: int = 2000
    seed: int = 0


def _reference_and_current(scenario: RetrainScenario):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(scenario.seed)
    x, y = linear_boundary_xy(rng, scenario.n)
    reference = pd.DataFrame(x, columns=_SYNTH_FEATURES)
    current = reference.copy()
    if scenario.inject_drift:
        current = SIM.covariate_shift(current, "f0", magnitude=1.0, delta=2.5)
    champion = ColumnSubsetModel(LogisticRegression().fit(x[:, :1], y), cols=[0])  # weak incumbent
    return reference, current, y, current.to_numpy(), champion


@asset(retry_policy=RetryPolicy(max_retries=2))
def drift_gated_retrain(
    context: AssetExecutionContext, retrain_scenario: RetrainScenario
) -> MaterializeResult:
    """Detect drift; retrain + gate ONLY if it fires (the conditional-retrain branch)."""
    from sklearn.linear_model import LogisticRegression

    reference, current, y, x_current, champion = _reference_and_current(retrain_scenario)
    report = DET.detect_drift(
        reference, current, numeric_features=_SYNTH_FEATURES, categorical_features=[]
    )
    triggered = report.covariate_shift or bool(report.drifted_features)
    if not triggered:
        context.log.info("no drift detected -> retrain skipped")
        return MaterializeResult(
            metadata={"drift_detected": False, "retrained": False, "promoted": False}
        )

    context.log.info(f"drift on {report.drifted_features} -> retraining challenger")
    outcome = retrain_and_gate(
        lambda: ColumnSubsetModel(
            LogisticRegression(max_iter=1000).fit(x_current, y), cols=[0, 1, 2]
        ),
        champion,
        x_current,
        y,
        segment_frame=None,
        n_rounds=200,
        seed=1,
    )
    return MaterializeResult(
        metadata={
            "drift_detected": True,
            "retrained": True,
            "promoted": bool(outcome.promoted),
            "drifted_features": ",".join(report.drifted_features),
            "domain_classifier_auc": float(report.domain_classifier_auc),
        }
    )
