"""Dagster assets wiring the Phase-2 slice: generate -> offline PIT -> online -> parity report.

The DAG orchestrates the BATCH lineage and makes online/offline parity a first-class, materialisable
asset (``parity_report`` fails the run if the streaming aggregation and the DuckDB PIT query
disagree). The streaming *transport* (Redpanda/Quix/Redis) is validated separately against live
infra in ``tests/test_streaming_integration.py``; here the broker-free reference aggregation
(:mod:`churn.streaming.aggregate`) stands in, so the whole slice materialises in-process and is a
fast, deterministic gate on the train/serve contract.

Assets read a :class:`SliceConfig` resource (seed / cohort size / query instant), so the same graph
drives tests and a real run without code edits.

(No ``from __future__ import annotations`` here: Dagster introspects the real annotation objects to
recognise the ``AssetExecutionContext`` parameter; stringised annotations would break that.)
"""

import pandas as pd
from dagster import AssetExecutionContext, ConfigurableResource, Failure, MaterializeResult, asset

_PARITY_TOL = 1e-9


class SliceConfig(ConfigurableResource):
    """Configuration for the Phase-2 slice (injected into every asset)."""

    seed: int = 4242
    n_synthetic: int = 0
    t0: str = "2025-01-01T00:00:00"
    n_customers: int = 40


@asset
def synthetic_dataset(slice_config: SliceConfig) -> dict:
    """Generate the frozen-world dataset and slice a cohort of customers that have events."""
    from churn.simulator import generate as G
    from churn.simulator import params as P

    ds = G.build_population_dataset(
        P.load_params(), seed=slice_config.seed, n_synthetic=slice_config.n_synthetic
    )
    counts = ds.events["customer_id"].value_counts()
    ids = [c for c in ds.customers["customer_id"].tolist() if counts.get(c, 0) > 0]
    ids = ids[: slice_config.n_customers]
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)
    return {"events": events, "customer_ids": ids}


@asset
def offline_pit_features(synthetic_dataset: dict, slice_config: SliceConfig) -> pd.DataFrame:
    """Offline DuckDB point-in-time features keyed by customer (the batch scoring path)."""
    from churn.featurestore import offline as OFF

    ids = synthetic_dataset["customer_ids"]
    t0 = pd.Timestamp(slice_config.t0)
    cohort = pd.DataFrame({"customer_id": ids, "t0": [t0] * len(ids)})
    return OFF.compute_pit_features(synthetic_dataset["events"], cohort).set_index("customer_id")


@asset
def online_features(synthetic_dataset: dict, slice_config: SliceConfig) -> pd.DataFrame:
    """Online event-time aggregation (broker-free reference impl = the consumer's logic)."""
    from churn.streaming import aggregate as AGG

    ids = synthetic_dataset["customer_ids"]
    t0 = pd.Timestamp(slice_config.t0)
    vectors = AGG.aggregate_stream(synthetic_dataset["events"], {c: t0 for c in ids})
    return pd.DataFrame.from_dict(vectors, orient="index")


@asset
def parity_report(
    context: AssetExecutionContext,
    offline_pit_features: pd.DataFrame,
    online_features: pd.DataFrame,
) -> MaterializeResult:
    """Assert online == offline across the cohort; fail the run (and record the diff) otherwise."""
    from churn.featurestore import offline as OFF

    cols = OFF.FEATURE_COLUMNS
    offline = offline_pit_features[cols].sort_index()
    online = online_features.reindex(offline.index)[cols]
    max_abs_diff = float((offline - online).abs().to_numpy().max())
    context.log.info(f"online/offline parity: max_abs_diff={max_abs_diff:.2e} over {len(offline)}")
    if max_abs_diff > _PARITY_TOL:
        raise Failure(f"online/offline parity broken: max_abs_diff={max_abs_diff:.3e}")
    return MaterializeResult(
        metadata={
            "max_abs_diff": max_abs_diff,
            "n_customers": len(offline),
            "parity_ok": True,
        }
    )
