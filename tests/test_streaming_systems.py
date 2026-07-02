"""Streaming systems demo against LIVE Redpanda + Redis -- Redpanda's "earn it" bar (Sec 4.4).

Beyond correctness, the broker must demonstrate real streaming guarantees: large-scale parity +
throughput, idempotent replay (at-least-once + dedup), and crash recovery. Each test drives real
infra; the invariant is always that the online store converges to the offline PIT vector.

Runs in the ``test-runner`` container: ``docker compose --profile test run --rm test-runner``.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("confluent_kafka")
pytest.importorskip("quixstreams")

import pandas as pd  # noqa: E402

from churn.featurestore import offline as OFF  # noqa: E402
from churn.featurestore.online import OnlineStore, RedisBackend  # noqa: E402
from churn.simulator import generate as G  # noqa: E402
from churn.simulator import params as P  # noqa: E402
from churn.streaming import producer as PROD  # noqa: E402
from services.consumer import app as CONSUMER  # noqa: E402

pytestmark = pytest.mark.integration

T0 = pd.Timestamp("2025-01-01T00:00:00")
# Conservative floor: demonstrates a measured throughput gate without laptop/CI flakiness.
_MIN_EVENTS_PER_SEC = 20.0


def _cohort(ds, n: int) -> list[str]:
    counts = ds.events["customer_id"].value_counts()
    have = [c for c in ds.customers["customer_id"].tolist() if counts.get(c, 0) > 0]
    return have[:n]


def _offline(events: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    cohort = pd.DataFrame({"customer_id": ids, "t0": [T0] * len(ids)})
    return OFF.compute_pit_features(events, cohort).set_index("customer_id")


def _assert_store_matches_offline(
    store: OnlineStore, offline: pd.DataFrame, ids: list[str]
) -> None:
    for c in ids:
        served = store.get(c)
        assert served is not None, f"no online features for {c}"
        for col in OFF.FEATURE_COLUMNS:
            assert served[col] == pytest.approx(float(offline.loc[c, col]), abs=1e-9), (c, col)


def test_large_scale_parity_and_throughput(
    redpanda_broker, redis_url, kafka_topic, unique_group, redis_client, tmp_path
):
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _cohort(ds, 60)  # ~29k events -- a genuinely large stream (O(1)/event consumer)
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)
    shuffled = events.sample(frac=1.0, random_state=3).reset_index(drop=True)

    produced = PROD.produce_events(redpanda_broker, kafka_topic, shuffled)
    store = OnlineStore(RedisBackend(redis_url))

    start = time.perf_counter()
    processed = CONSUMER.run_consumer(
        broker=redpanda_broker,
        topic=kafka_topic,
        consumer_group=unique_group,
        t0=T0,
        online_store=store,
        timeout=30.0,
        count=produced,  # stop exactly at the batch size -> clean throughput measurement
        state_dir=str(tmp_path / "quix-state"),
    )
    elapsed = time.perf_counter() - start
    rate = processed / elapsed

    assert processed == produced
    print(f"throughput: {processed} events in {elapsed:.1f}s = {rate:.0f} events/s")
    assert rate > _MIN_EVENTS_PER_SEC, f"throughput {rate:.0f}/s below {_MIN_EVENTS_PER_SEC}/s"
    _assert_store_matches_offline(store, _offline(events, ids), ids)


def test_fresh_group_replay_is_idempotent(
    redpanda_broker, redis_url, kafka_topic, redis_client, tmp_path
):
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _cohort(ds, 20)
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)
    # Duplicates in the stream too: at-least-once redelivery must not move a feature.
    adversarial = pd.concat([events, events.iloc[:25]], ignore_index=True)
    PROD.produce_events(redpanda_broker, kafka_topic, adversarial)

    store = OnlineStore(RedisBackend(redis_url))

    def _drain(group: str, sub: str) -> dict:
        CONSUMER.run_consumer(
            broker=redpanda_broker,
            topic=kafka_topic,
            consumer_group=group,
            t0=T0,
            online_store=store,
            timeout=10.0,
            state_dir=str(tmp_path / sub),
        )
        return {c: store.get(c) for c in ids}

    first = _drain("replay-a", "state-a")
    redis_client.flushdb()
    second = _drain("replay-b", "state-b")  # fresh group replays from the beginning

    assert first == second, "replay from offset 0 did not reproduce the identical online state"
    _assert_store_matches_offline(store, _offline(events, ids), ids)


def test_crash_recovery_converges(
    redpanda_broker, redis_url, kafka_topic, unique_group, redis_client, tmp_path
):
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _cohort(ds, 30)
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)
    produced = PROD.produce_events(redpanda_broker, kafka_topic, events)

    store = OnlineStore(RedisBackend(redis_url))
    state_dir = str(tmp_path / "quix-state")
    common = dict(
        broker=redpanda_broker,
        topic=kafka_topic,
        consumer_group=unique_group,
        t0=T0,
        online_store=store,
        commit_every=1,  # persist offsets so the restart resumes rather than restarts
        state_dir=state_dir,
    )

    # "Crash" mid-stream: stop after half the batch, then restart the same group + state.
    partial = CONSUMER.run_consumer(**common, timeout=30.0, count=produced // 2)
    assert partial == produced // 2
    CONSUMER.run_consumer(**common, timeout=10.0)  # resume to completion

    _assert_store_matches_offline(store, _offline(events, ids), ids)
