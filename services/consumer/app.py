"""Quix Streams consumer -- Redpanda topic -> per-customer event-time state -> Redis online store.

Deployment glue around the parity-proven online reference implementation
(:mod:`churn.streaming.aggregate`). Quix owns transport + per-key stateful storage; the state
payload is the deduped event map and :func:`churn.streaming.aggregate.feature_vector` is the
reducer, so the DEPLOYED consumer and the offline DuckDB PIT query stay byte-identical -- the
platform's train/serve-consistency guarantee, proven end-to-end in
``tests/test_streaming_integration.py`` against a LIVE broker + store.

Features are materialised AS OF a fixed query instant ``t0`` (``--as-of`` / ``CHURN_AS_OF``): each
event with ``event_ts <= t0`` folds into the per-customer state and the current feature vector is
written to Redis on every update. Because :func:`feature_vector` is a pure function of the deduped
event set and ``t0``, late / duplicate / reordered delivery converges to the exact offline vector.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from churn.featurestore.online import OnlineStore
from churn.streaming import aggregate as AGG

DEFAULT_TOPIC = "customer-events"
DEFAULT_GROUP = "churn-streaming-v1"


def _ts_extractor(value: dict, headers: object, timestamp: int, timestamp_type: object) -> int:
    # Event time (ms) drives Quix's timeline; the feature math uses the ns from the payload.
    return int(value["event_ts_ns"]) // 1_000_000


def build_online_store(redis_url: str) -> OnlineStore:
    return OnlineStore.from_url(redis_url)


def run_consumer(
    *,
    broker: str,
    topic: str,
    consumer_group: str,
    t0: pd.Timestamp,
    online_store: OnlineStore,
    timeout: float = 10.0,
    count: int = 0,
    commit_every: int = 0,
    state_dir: str | None = None,
    auto_offset_reset: str = "earliest",
    use_changelog_topics: bool = True,
) -> int:
    """Consume ``topic`` and write each customer's online vector to ``online_store``.

    The run is bounded by ``timeout`` (stop after this many seconds with no new message; 0 = run
    forever) and/or ``count`` (stop after this many messages; 0 = unbounded) -- whichever hits
    first. ``commit_every`` forces an offset commit every N messages (0 = interval-based). Returns
    the number of events processed. Reuses the SAME reducer as the offline PIT query, so what lands
    in Redis is the offline feature vector by construction; dedup-by-event_id makes reprocessing
    (at-least-once redelivery / crash recovery) idempotent.

    ``use_changelog_topics`` (default True) is load-bearing for that guarantee across a restart.
    Quix checkpoints in the order *produce state to the changelog -> commit input offsets -> flush
    local state to disk*, so committed offsets can outrun the on-disk state. Without the changelog,
    any restart that loses the local ``state_dir`` (an ephemeral container, or a crash in the
    commit->flush window) resumes past already-committed events with EMPTY per-customer state --
    those events are never redelivered, so ``CustomerAggregator`` permanently undercounts and Redis
    diverges from the offline PIT vector. With it, Quix rebuilds local state from the durable
    changelog (written before the offset commit) before resuming, so parity survives a full state
    wipe. Kept as a parameter so the regression test can exercise both paths.
    """
    from quixstreams import Application
    from quixstreams.state import State

    t0 = pd.Timestamp(t0)
    state_dir = state_dir or os.getenv("CHURN_STATE_DIR", "/tmp/quix-state")
    app = Application(
        broker_address=broker,
        consumer_group=consumer_group,
        auto_offset_reset=auto_offset_reset,
        state_dir=state_dir,
        use_changelog_topics=use_changelog_topics,
        commit_every=commit_every,
    )
    in_topic = app.topic(
        topic,
        value_deserializer="json",
        key_deserializer="str",
        timestamp_extractor=_ts_extractor,
    )
    sdf = app.dataframe(topic=in_topic)
    t0_ns = int(t0.value)
    counter = {"n": 0}

    def update_and_write(event: dict, state: State) -> dict:
        # Count every consumed message; dedup by event_id via a per-key marker so at-least-once
        # redelivery / crash-recovery reprocessing is idempotent (O(1) point lookups, no growing
        # blob). New events fold into a fixed-size incremental aggregator -- O(1) per event.
        counter["n"] += 1
        eid = str(event["event_id"])
        if state.exists(f"s:{eid}"):
            return event
        state.set(f"s:{eid}", 1)
        raw = state.get("agg")
        agg = (
            AGG.CustomerAggregator.from_state(json.loads(raw))
            if raw
            else AGG.CustomerAggregator(t0_ns=t0_ns)
        )
        agg.add(int(event["event_ts_ns"]), str(event["event_type"]), event.get("value"))
        state.set("agg", json.dumps(agg.to_state()))
        online_store.put(str(event["customer_id"]), agg.features())
        return event

    sdf = sdf.apply(update_and_write, stateful=True)
    app.run(timeout=timeout, count=count)
    return counter["n"]


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the compose ``consumer`` service."""
    parser = argparse.ArgumentParser(description="Stream events -> Redis online features.")
    parser.add_argument("--broker", default=os.getenv("REDPANDA_BROKER", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("EVENTS_TOPIC", DEFAULT_TOPIC))
    parser.add_argument("--group", default=os.getenv("CONSUMER_GROUP", DEFAULT_GROUP))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--as-of", default=os.getenv("CHURN_AS_OF", "2025-01-01T00:00:00"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("CONSUMER_TIMEOUT", "0")))
    args = parser.parse_args(argv)

    store = build_online_store(args.redis_url)
    processed = run_consumer(
        broker=args.broker,
        topic=args.topic,
        consumer_group=args.group,
        t0=pd.Timestamp(args.as_of),
        online_store=store,
        timeout=args.timeout,
    )
    print(f"Consumer stopped after {processed:,} events (topic={args.topic}, group={args.group})")


if __name__ == "__main__":
    main()
