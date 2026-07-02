"""Event-time streaming aggregation (the Quix Streams logic, broker-free + testable).

An **independent** Python implementation of the online feature state: process events one at a
time in arbitrary arrival order, deduplicate by ``event_id`` (idempotent), keep per-customer
event-time state, and emit -- at a query instant ``t0`` -- the byte-identical feature vector the
offline DuckDB PIT query produces (``featurestore.offline``). Because it is a *separate*
implementation (Python vs SQL), the parity test (``tests/test_streaming_parity.py``) is a real
cross-check, not a tautology: late / duplicate / reordered / boundary events must not move a
feature. The column set + windows are read from ``offline`` (single source), but every value is
computed here from scratch.

In production this logic is the Quix Streams consumer, keyed by ``customer_id``: ``CustomerState``
is the per-key state (serialisable to a plain dict via ``as_dict``/``from_dict`` so Quix can
persist it), and ``feature_vector`` is the reducer the consumer writes to Redis (the online
store). ``services.consumer`` imports these directly, so there is exactly ONE online reference
implementation behind both the parity test and the deployed consumer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from churn.featurestore import offline as OFF

_SPEC = OFF.DEFAULT_FEATURE_SPEC
_W = _SPEC["windows"]
_SENTINEL = _SPEC["recency_sentinel_days"]

# One event as held in state: (event_time, event_type, value). ``value`` is NaN for event types
# that carry no magnitude (order / payment_fail / downgrade -- only their counts matter).
EventRow = tuple[pd.Timestamp, str, float]


def _as_float(value: object) -> float:
    # None (JSON null over the wire) and NaN both collapse to NaN; everything else is a float.
    if value is None:
        return float("nan")
    v = float(value)
    return v


class CustomerState:
    """Per-customer deduped, event-time event store (the online aggregation state).

    Dedup is by ``event_id`` (idempotent): a redelivered duplicate overwrites with the same
    payload, so late / duplicate delivery can never move a feature. The state round-trips through
    ``as_dict``/``from_dict`` (event time as int64 nanoseconds) so the Quix consumer can persist
    and recover it as keyed state.
    """

    __slots__ = ("_events",)

    def __init__(self, events: Mapping[str, EventRow] | None = None) -> None:
        self._events: dict[str, EventRow] = dict(events) if events else {}

    def update(self, event_id: str, ts: object, event_type: str, value: object) -> None:
        self._events[event_id] = (pd.Timestamp(ts), event_type, _as_float(value))

    def features(self, t0: pd.Timestamp) -> dict[str, float]:
        return feature_vector(self._events, pd.Timestamp(t0))

    def as_dict(self) -> dict[str, tuple[int, str, float]]:
        """JSON/state-friendly view: event time as int64 ns (lossless, timezone-free)."""
        return {eid: (ts.value, typ, val) for eid, (ts, typ, val) in self._events.items()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Sequence]) -> CustomerState:
        events = {
            eid: (pd.Timestamp(int(ts)), typ, _as_float(val)) for eid, (ts, typ, val) in raw.items()
        }
        return cls(events)


def _in_window(ts: pd.Timestamp, t0: pd.Timestamp, days: int) -> bool:
    # Matches the offline SQL: event_ts > t0 - INTERVAL {days} DAY (strict lower bound).
    return ts > t0 - pd.Timedelta(days=days)


def _win_count(timestamps: list[pd.Timestamp], t0: pd.Timestamp, days: int) -> float:
    return float(sum(1 for ts in timestamps if _in_window(ts, t0, days)))


def _mean_or_sentinel(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_features(rows: list[EventRow], t0: pd.Timestamp) -> dict[str, float]:
    """Reduce PIT-filtered ``(ts, type, value)`` rows to the frozen feature vector at ``t0``.

    ``rows`` must already be restricted to ``event_ts <= t0`` (the label window is invisible).
    """
    logins = [(ts, val) for ts, typ, val in rows if typ == "login"]
    supports = [(ts, val) for ts, typ, val in rows if typ == "support"]
    login_ts = [ts for ts, _ in logins]
    support_ts = [ts for ts, _ in supports]
    order_ts = [ts for ts, typ, _ in rows if typ == "order"]
    payment_ts = [ts for ts, typ, _ in rows if typ == "payment_fail"]
    downgrade_ts = [ts for ts, typ, _ in rows if typ == "downgrade"]

    login_28 = _win_count(login_ts, t0, _W["short_days"])
    login_90 = _win_count(login_ts, t0, _W["mid_days"])
    support_90 = _win_count(support_ts, t0, _W["mid_days"])
    sd28 = [v for ts, v in logins if _in_window(ts, t0, _W["short_days"])]
    sd90 = [v for ts, v in logins if _in_window(ts, t0, _W["mid_days"])]
    sent90 = [v for ts, v in supports if _in_window(ts, t0, _W["mid_days"])]

    last_login = max(login_ts, default=None)
    recency = (
        _SENTINEL
        if last_login is None
        else min(_SENTINEL, (t0 - last_login).total_seconds() / 86_400.0)
    )

    return {
        "days_since_last_login": float(recency),
        "login_count_14d": _win_count(login_ts, t0, _W["near_days"]),
        "login_count_28d": login_28,
        "login_count_90d": login_90,
        "login_trend_28_90": float(login_28 / (login_90 + 1.0)),
        "mean_session_depth_28d": _mean_or_sentinel(sd28),
        "mean_session_depth_90d": _mean_or_sentinel(sd90),
        "has_login_90d": 1.0 if login_90 > 0 else 0.0,
        "order_count_90d": _win_count(order_ts, t0, _W["mid_days"]),
        "support_count_90d": support_90,
        "mean_sentiment_90d": _mean_or_sentinel(sent90),
        "has_support_90d": 1.0 if support_90 > 0 else 0.0,
        "payment_fail_count_180d": _win_count(payment_ts, t0, _W["pay_days"]),
        "downgrade_count_365d": _win_count(downgrade_ts, t0, _W["full_days"]),
    }


def feature_vector(events_by_id: Mapping[str, EventRow], t0: pd.Timestamp) -> dict[str, float]:
    """One customer's feature vector from a deduped event map at ``t0`` (the recompute reducer).

    This is the reference reducer: correct and obvious (recompute from the full deduped set), but
    O(events) per call. The deployed consumer uses :class:`CustomerAggregator` (O(1) per event); a
    parity test pins the two equal, so the fast path inherits this one's correctness.
    """
    t0 = pd.Timestamp(t0)
    rows = [
        (pd.Timestamp(ts), typ, _as_float(val))
        for ts, typ, val in events_by_id.values()
        if pd.Timestamp(ts) <= t0
    ]
    return compute_features(rows, t0)


_NS_PER_DAY = 86_400_000_000_000  # 86_400 s * 1e9 ns; window bounds in event-time nanoseconds


@dataclass
class CustomerAggregator:
    """Incremental, O(1)-per-event online aggregation for a FIXED query instant ``t0``.

    Window membership is static for a fixed ``t0`` (an event is in the N-day window iff
    ``event_ts > t0 - N days``), so each new event updates a constant set of running counters --
    no per-event recompute over history. The state is a fixed-size, JSON-friendly blob
    (``to_state``/``from_state``) suitable for a streaming consumer's keyed state. Dedup-by-event_id
    is the caller's responsibility (the consumer skips already-seen ids), so ``add`` assumes each
    event is applied once. ``features()`` reproduces :func:`compute_features` exactly.
    """

    t0_ns: int
    login_14: int = 0
    login_28: int = 0
    login_90: int = 0
    sd28_sum: float = 0.0
    sd28_n: int = 0
    sd90_sum: float = 0.0
    sd90_n: int = 0
    support_90: int = 0
    sent90_sum: float = 0.0
    sent90_n: int = 0
    order_90: int = 0
    pay_180: int = 0
    down_365: int = 0
    last_login_ns: int | None = None

    def add(self, ts_ns: int, event_type: str, value: object) -> None:
        t0 = self.t0_ns
        if ts_ns > t0:  # strict PIT: the label window is invisible
            return
        if event_type == "login":
            val = _as_float(value)
            if ts_ns > t0 - 14 * _NS_PER_DAY:
                self.login_14 += 1
            if ts_ns > t0 - 28 * _NS_PER_DAY:
                self.login_28 += 1
                self.sd28_sum += val
                self.sd28_n += 1
            if ts_ns > t0 - 90 * _NS_PER_DAY:
                self.login_90 += 1
                self.sd90_sum += val
                self.sd90_n += 1
            if self.last_login_ns is None or ts_ns > self.last_login_ns:
                self.last_login_ns = ts_ns
        elif event_type == "support":
            if ts_ns > t0 - 90 * _NS_PER_DAY:
                self.support_90 += 1
                self.sent90_sum += _as_float(value)
                self.sent90_n += 1
        elif event_type == "order":
            if ts_ns > t0 - 90 * _NS_PER_DAY:
                self.order_90 += 1
        elif event_type == "payment_fail":
            if ts_ns > t0 - 180 * _NS_PER_DAY:
                self.pay_180 += 1
        elif event_type == "downgrade":
            if ts_ns > t0 - 365 * _NS_PER_DAY:
                self.down_365 += 1

    def features(self) -> dict[str, float]:
        t0 = pd.Timestamp(self.t0_ns)
        if self.last_login_ns is None:
            recency = _SENTINEL
        else:
            elapsed = (t0 - pd.Timestamp(self.last_login_ns)).total_seconds() / 86_400.0
            recency = min(_SENTINEL, elapsed)
        return {
            "days_since_last_login": float(recency),
            "login_count_14d": float(self.login_14),
            "login_count_28d": float(self.login_28),
            "login_count_90d": float(self.login_90),
            "login_trend_28_90": float(self.login_28 / (self.login_90 + 1.0)),
            "mean_session_depth_28d": self.sd28_sum / self.sd28_n if self.sd28_n else 0.0,
            "mean_session_depth_90d": self.sd90_sum / self.sd90_n if self.sd90_n else 0.0,
            "has_login_90d": 1.0 if self.login_90 > 0 else 0.0,
            "order_count_90d": float(self.order_90),
            "support_count_90d": float(self.support_90),
            "mean_sentiment_90d": self.sent90_sum / self.sent90_n if self.sent90_n else 0.0,
            "has_support_90d": 1.0 if self.support_90 > 0 else 0.0,
            "payment_fail_count_180d": float(self.pay_180),
            "downgrade_count_365d": float(self.down_365),
        }

    def to_state(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, raw: Mapping[str, object]) -> CustomerAggregator:
        return cls(**raw)


def feature_vector_incremental(
    events_by_id: Mapping[str, EventRow], t0: pd.Timestamp
) -> dict[str, float]:
    """Fold a deduped event map through :class:`CustomerAggregator` (the fast-path reducer)."""
    agg = CustomerAggregator(t0_ns=int(pd.Timestamp(t0).value))
    for ts, typ, val in events_by_id.values():
        agg.add(int(pd.Timestamp(ts).value), typ, val)
    return agg.features()


def _record_field(rec: object, name: str) -> object:
    return rec[name] if isinstance(rec, dict) else getattr(rec, name)


def aggregate_stream(
    events: object, query_times: Mapping[str, pd.Timestamp]
) -> dict[str, dict[str, float]]:
    """Fold an event stream (any order) into per-customer online vectors at ``query_times``.

    ``events`` is any iterable of rows with ``event_id, customer_id, event_ts, event_type,
    value`` (a DataFrame or list of dicts). Returns ``{customer_id: feature_vector}`` for every
    requested customer, emitting the empty-window sentinels for a customer with no events.
    """
    states: dict[str, CustomerState] = {c: CustomerState() for c in query_times}
    records = events.itertuples(index=False) if isinstance(events, pd.DataFrame) else events
    for rec in records:
        cid = _record_field(rec, "customer_id")
        if cid not in states:
            continue
        states[cid].update(
            _record_field(rec, "event_id"),
            _record_field(rec, "event_ts"),
            _record_field(rec, "event_type"),
            _record_field(rec, "value"),
        )
    return {c: states[c].features(pd.Timestamp(t0)) for c, t0 in query_times.items()}


def aggregate_stream_incremental(
    events: object, query_times: Mapping[str, pd.Timestamp]
) -> dict[str, dict[str, float]]:
    """Same result as :func:`aggregate_stream`, via the O(1)-per-event incremental path.

    Mirrors the deployed consumer exactly (dedup by ``event_id`` then fold through
    :class:`CustomerAggregator`), so the parity test exercises the real online algorithm.
    """
    aggs = {
        c: CustomerAggregator(t0_ns=int(pd.Timestamp(t0).value)) for c, t0 in query_times.items()
    }
    seen: dict[str, set[str]] = {c: set() for c in query_times}
    records = events.itertuples(index=False) if isinstance(events, pd.DataFrame) else events
    for rec in records:
        cid = _record_field(rec, "customer_id")
        if cid not in aggs:
            continue
        eid = _record_field(rec, "event_id")
        if eid in seen[cid]:
            continue
        seen[cid].add(eid)
        aggs[cid].add(
            int(pd.Timestamp(_record_field(rec, "event_ts")).value),
            _record_field(rec, "event_type"),
            _record_field(rec, "value"),
        )
    return {c: aggs[c].features() for c in query_times}
