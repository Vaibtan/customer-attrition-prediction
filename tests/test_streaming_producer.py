"""Producer wire-format unit tests (pure -- no broker, runs on the host).

The wire format is the train/serve contract's first link: event time must survive JSON transport
losslessly (int64 nanoseconds) and a missing magnitude (NaN) must round-trip as JSON null so the
online aggregation reconstructs the exact same event set the offline PIT query sees.
"""

from __future__ import annotations

import math
import sys
import types

import pandas as pd
import pytest

from churn.streaming import producer as PROD

T0 = pd.Timestamp("2025-01-01T00:00:00")


def test_serialize_round_trips_a_login_event_losslessly():
    rec = {
        "event_id": "A:login:0",
        "customer_id": "A",
        "event_ts": T0 - pd.Timedelta(days=3, minutes=17, seconds=42),
        "event_type": "login",
        "value": 6.0,
    }
    wire = PROD.deserialize_event(PROD.serialize_event(rec))
    assert wire["event_id"] == "A:login:0"
    assert wire["customer_id"] == "A"
    assert wire["event_type"] == "login"
    assert wire["value"] == 6.0
    # Event time survives as int64 ns and reconstructs the exact original instant.
    assert isinstance(wire["event_ts_ns"], int)
    assert pd.Timestamp(wire["event_ts_ns"]) == rec["event_ts"]


def test_nan_value_serialises_to_null():
    rec = {
        "event_id": "A:order:0",
        "customer_id": "A",
        "event_ts": T0,
        "event_type": "order",
        "value": float("nan"),
    }
    wire = PROD.deserialize_event(PROD.serialize_event(rec))
    assert wire["value"] is None


def test_serialize_is_deterministic_bytes():
    rec = {
        "event_id": "A:sup:0",
        "customer_id": "A",
        "event_ts": T0,
        "event_type": "support",
        "value": -0.4,
    }
    assert PROD.serialize_event(rec) == PROD.serialize_event(dict(rec))


def test_iter_records_normalises_a_dataframe():
    df = pd.DataFrame(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0,
                "event_type": "login",
                "value": 5.0,
            },
            {
                "event_id": "B:order:0",
                "customer_id": "B",
                "event_ts": T0 - pd.Timedelta(days=1),
                "event_type": "order",
                "value": math.nan,
            },
        ]
    )
    records = list(PROD.iter_records(df))
    assert len(records) == 2
    assert {r["customer_id"] for r in records} == {"A", "B"}
    assert all(set(PROD.EVENT_FIELDS) <= set(PROD.to_wire(r)) for r in records)


# --- delivery semantics (REV-22): acked count, BufferError backpressure, failure raises ----------


def _events(n: int) -> list[dict]:
    return [
        {
            "event_id": f"A:login:{i}",
            "customer_id": "A",
            "event_ts": T0 - pd.Timedelta(hours=i),
            "event_type": "login",
            "value": float(i),
        }
        for i in range(n)
    ]


def _install_fake_kafka(monkeypatch, *, buffer_rejects: int = 0, fail_indices: tuple = ()):
    """Install a fake ``confluent_kafka`` whose Producer queues delivery callbacks.

    Callbacks are served only on blocking ``poll`` / ``flush`` (the worst case), so the acked
    count genuinely depends on the delivery reports, not the enqueue count. ``buffer_rejects``
    makes the first N ``produce`` calls raise ``BufferError``; ``fail_indices`` marks messages
    (by enqueue order) whose delivery report carries an error.
    """
    instances: list = []

    class FakeProducer:
        def __init__(self, conf):
            self.messages: list = []
            self.pending: list = []
            self.polls: list[float] = []
            self._rejects_left = buffer_rejects
            instances.append(self)

        def produce(self, topic, key=None, value=None, on_delivery=None):
            if self._rejects_left > 0:
                self._rejects_left -= 1
                raise BufferError("Local: Queue full")
            i = len(self.messages)
            self.messages.append((topic, key, value))
            err = "Broker: Message timed out" if i in fail_indices else None
            self.pending.append((on_delivery, err))

        def poll(self, timeout=0):
            self.polls.append(timeout)
            if timeout > 0:  # blocking poll frees queue space by serving reports
                self._serve()

        def flush(self):
            self._serve()

        def _serve(self):
            while self.pending:
                cb, err = self.pending.pop(0)
                cb(err, object())

    mod = types.ModuleType("confluent_kafka")
    mod.Producer = FakeProducer
    monkeypatch.setitem(sys.modules, "confluent_kafka", mod)
    return instances


def test_produce_events_returns_the_acknowledged_count(monkeypatch):
    instances = _install_fake_kafka(monkeypatch)
    assert PROD.produce_events("broker:9092", "t", _events(5)) == 5
    assert len(instances[0].messages) == 5  # everything actually reached produce()


def test_produce_events_survives_a_full_local_queue(monkeypatch):
    instances = _install_fake_kafka(monkeypatch, buffer_rejects=3)
    assert PROD.produce_events("broker:9092", "t", _events(4)) == 4
    prod = instances[0]
    assert len(prod.messages) == 4  # the rejected produces were retried, not dropped
    assert any(t > 0 for t in prod.polls)  # backpressure was a blocking poll, not a crash


def test_produce_events_raises_when_a_delivery_fails(monkeypatch):
    _install_fake_kafka(monkeypatch, fail_indices=(1,))
    with pytest.raises(RuntimeError, match="1/3 events failed delivery"):
        PROD.produce_events("broker:9092", "t", _events(3))


def test_produce_events_without_flush_reports_enqueued_only(monkeypatch):
    instances = _install_fake_kafka(monkeypatch)
    assert PROD.produce_events("broker:9092", "t", _events(2), flush=False) == 2
    assert instances[0].pending  # callbacks still outstanding: the count is enqueued, not acked
