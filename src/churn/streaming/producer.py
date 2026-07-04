"""Redpanda producer -- replay the event log to a Kafka topic (confluent-kafka).

``to_wire`` / ``serialize_event`` / ``deserialize_event`` / ``iter_records`` are pure and
host-testable; ``produce_events`` opens a real ``confluent_kafka.Producer`` (imported lazily so the
base install and Windows dev box stay broker-free) and replays a frame/iterable, keyed by
``customer_id`` so every customer's events land on one partition -- the consumer's per-customer
state is then partition-local. Event time is carried as int64 **nanoseconds** (lossless,
timezone-free) so the online aggregation reconstructs the exact ``pd.Timestamp``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pandas as pd

# The wire schema (JSON). Event time is int64 ns; ``value`` is a float or null (missing magnitude).
EVENT_FIELDS: tuple[str, ...] = ("event_id", "customer_id", "event_ts_ns", "event_type", "value")


def _clean_value(value: object) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if math.isnan(v) else v


def to_wire(record: Mapping[str, object]) -> dict[str, object]:
    """Normalise one event record ``{..., event_ts: Timestamp, value: float|NaN}`` to wire form."""
    ts = pd.Timestamp(record["event_ts"])
    return {
        "event_id": str(record["event_id"]),
        "customer_id": str(record["customer_id"]),
        "event_ts_ns": int(ts.value),
        "event_type": str(record["event_type"]),
        "value": _clean_value(record.get("value")),
    }


def _dumps_wire(wire: Mapping[str, object]) -> bytes:
    """Wire dict -> compact, deterministic JSON bytes (stable key order): the single wire codec."""
    return json.dumps(wire, separators=(",", ":"), sort_keys=True).encode("utf-8")


def serialize_event(record: Mapping[str, object]) -> bytes:
    """Event record -> compact, deterministic JSON bytes (stable key order)."""
    return _dumps_wire(to_wire(record))


def deserialize_event(raw: bytes | str) -> dict[str, object]:
    """Wire bytes -> the wire dict (``event_ts_ns`` int, ``value`` float|None)."""
    return json.loads(raw)


def iter_records(events: pd.DataFrame | Iterable[Mapping[str, object]]) -> Iterator[dict]:
    """Yield event record dicts from a DataFrame or any iterable of mappings."""
    if isinstance(events, pd.DataFrame):
        yield from events.to_dict("records")
    else:
        for rec in events:
            yield dict(rec)


def produce_events(
    broker: str,
    topic: str,
    events: pd.DataFrame | Iterable[Mapping[str, object]],
    *,
    flush: bool = True,
) -> int:
    """Replay ``events`` to ``topic`` on ``broker``, keyed by ``customer_id``. Returns the count."""
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": broker})
    count = 0
    for record in iter_records(events):
        wire = to_wire(record)
        producer.produce(
            topic,
            key=wire["customer_id"].encode("utf-8"),
            value=_dumps_wire(wire),
        )
        count += 1
        producer.poll(0)  # serve delivery callbacks without blocking
    if flush:
        producer.flush()
    return count


def main(argv: list[str] | None = None) -> None:
    """CLI for the compose ``producer`` service: replay a parquet event log to a topic."""
    parser = argparse.ArgumentParser(description="Replay an event log to a Redpanda topic.")
    parser.add_argument("--broker", default=os.getenv("REDPANDA_BROKER", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("EVENTS_TOPIC", "customer-events"))
    default_events = os.getenv("EVENTS_PARQUET", "data/stream/events.parquet")
    parser.add_argument("--events", default=default_events)
    args = parser.parse_args(argv)

    events = pd.read_parquet(Path(args.events))
    n = produce_events(args.broker, args.topic, events)
    print(f"Produced {n:,} events -> {args.topic} @ {args.broker}")


if __name__ == "__main__":
    main()
