"""End-to-end streaming parity against LIVE Redpanda + Redis (no mocks).

The Phase-2 deliverable, proven through the ACTUAL deployment path: produce the event log to a real
Redpanda topic, run the real Quix Streams consumer, read the online feature store from real Redis,
and assert every customer's online vector == the offline DuckDB PIT vector -- under adversarial
delivery (reordered + duplicated). This is the train/serve-consistency guarantee the platform
exists to provide, exercised against live infra rather than an in-process fake.

Runs only where the infra + Linux clients exist (the ``test-runner`` container); skipped on the
host. Invoke: ``docker compose --profile test run --rm test-runner``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("confluent_kafka")
pytest.importorskip("quixstreams")

import pandas as pd  # noqa: E402 -- after importorskip so host collection skips cleanly

from churn.featurestore import offline as OFF  # noqa: E402
from churn.featurestore.cohort import make_cohort  # noqa: E402
from churn.featurestore.online import OnlineStore  # noqa: E402
from churn.simulator import generate as G  # noqa: E402
from churn.simulator import params as P  # noqa: E402
from churn.streaming import producer as PROD  # noqa: E402
from services.consumer import app as CONSUMER  # noqa: E402

pytestmark = pytest.mark.integration

T0 = pd.Timestamp("2025-01-01T00:00:00")


def _customers_with_events(ds, n: int) -> list[str]:
    counts = ds.events["customer_id"].value_counts()
    have = [c for c in ds.customers["customer_id"].tolist() if counts.get(c, 0) > 0]
    return have[:n]


def test_streaming_parity_end_to_end(
    redpanda_broker, redis_url, kafka_topic, unique_group, redis_client, tmp_path
):
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _customers_with_events(ds, 25)
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)

    # Adversarial delivery: shuffle arrival order (arrival != event time) and redeliver a slice
    # (duplicates). Idempotent dedup-by-event_id + strict PIT must make the result identical.
    shuffled = events.sample(frac=1.0, random_state=1).reset_index(drop=True)
    adversarial = pd.concat([shuffled, shuffled.iloc[:15]], ignore_index=True)

    produced = PROD.produce_events(redpanda_broker, kafka_topic, adversarial)
    assert produced == len(adversarial)

    store = OnlineStore.from_url(redis_url)
    processed = CONSUMER.run_consumer(
        broker=redpanda_broker,
        topic=kafka_topic,
        consumer_group=unique_group,
        t0=T0,
        online_store=store,
        timeout=10.0,
        state_dir=str(tmp_path / "quix-state"),
    )
    assert processed == len(adversarial)

    cohort = make_cohort(ids, T0)
    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")

    for c in ids:
        envelope = store.get(c)
        assert envelope is not None, f"no online features for {c}"
        for col in OFF.FEATURE_COLUMNS:
            assert envelope.features[col] == pytest.approx(float(offline.loc[c, col]), abs=1e-9), (
                c,
                col,
            )
