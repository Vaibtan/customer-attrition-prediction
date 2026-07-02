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

from churn.featurestore.online import OnlineStore, RedisBackend
from churn.streaming import aggregate as AGG

DEFAULT_TOPIC = "customer-events"
DEFAULT_GROUP = "churn-streaming-v1"


def _ts_extractor(value: dict, headers: object, timestamp: int, timestamp_type: object) -> int:
    # Event time (ms) drives Quix's timeline; the feature math uses the ns from the payload.
    return int(value["event_ts_ns"]) // 1_000_000


def build_online_store(redis_url: str) -> OnlineStore:
    return OnlineStore(RedisBackend(redis_url))


def run_consumer(
    *,
    broker: str,
    topic: str,
    consumer_group: str,
    t0: pd.Timestamp,
    online_store: OnlineStore,
    timeout: float = 10.0,
    state_dir: str | None = None,
    auto_offset_reset: str = "earliest",
) -> int:
    """Consume ``topic`` and write each customer's online vector to ``online_store``.

    ``timeout`` bounds the run: it stops once no new message arrives within that many seconds
    (0 = run forever). Returns the number of events processed. Reuses the SAME reducer as the
    offline PIT query, so what lands in Redis is the offline feature vector by construction.
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
        use_changelog_topics=False,
    )
    in_topic = app.topic(
        topic,
        value_deserializer="json",
        key_deserializer="str",
        timestamp_extractor=_ts_extractor,
    )
    sdf = app.dataframe(topic=in_topic)
    counter = {"n": 0}

    def update_and_write(event: dict, state: State) -> dict:
        # Per-key (customer) state: the deduped event map {event_id: [ts_ns, type, value]}.
        raw = state.get("events")
        events = json.loads(raw) if raw else {}
        events[str(event["event_id"])] = [
            int(event["event_ts_ns"]),
            str(event["event_type"]),
            event.get("value"),
        ]
        state.set("events", json.dumps(events))
        online_store.put(str(event["customer_id"]), AGG.feature_vector(events, t0))
        counter["n"] += 1
        return event

    sdf = sdf.apply(update_and_write, stateful=True)
    app.run(timeout=timeout)
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
