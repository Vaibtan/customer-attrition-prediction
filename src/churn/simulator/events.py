"""Event-stream generator (D6.1/D6.2/D6.6/D6.7/D6.8) -- draws the health-driven event log.

Given per-customer weekly health paths (the common cause), this emits the long, transport-faithful
event log ``events(event_id, customer_id, event_ts, event_type, value)`` that Phase 1 aggregates
into PIT features and Phase 2 replays into Redpanda. It composes the frozen event *links*
(``exp_link``/``logistic_link``/``linear_link``); the *draw families* (Exponential-arrival Poisson
counts / Bernoulli / Normal) are the D5.5/D5.10-declared ones; the *implementation* (timestamp
placement, 4-week cycle cadence, clipping) is Phase-1 code (spec section 5 boundary).

Invariants (D6): full 52-week emission for every customer regardless of tenure (D6.8); every event
strictly in ``(t0-364d, t0)`` (PIT); a per-cycle Bernoulli event's existence depends only on the
health of the week it is placed in (D6.2 placement-week evaluation); the whole draw is a pure
function of ``(params, seed)`` via the independent ``rng`` streams (so the null-stream control
regenerates only events, leaving health + label fixed -- D6.8 control-validity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from churn.simulator import kernels as K

# One shared UTC snapshot instant for the anchor cohort (D6.7). tz-naive UTC; the 52-week window is
# [ANCHOR_T0 - 364d, ANCHOR_T0). Pinned here so offline PIT and online streaming share one clock.
ANCHOR_T0 = pd.Timestamp("2025-01-01T00:00:00")
_DAY_NS = 86_400_000_000_000  # nanoseconds per day
_WEEK_NS = 7 * _DAY_NS
CYCLE_WEEKS = 4  # billing cadence for the per-cycle Bernoulli families (D6.2)

# event_type -> (link kind, params key, rng stream, payload family or None)
_RATE_FAMILIES = {
    "login": {"params": "login_rate", "stream": "login", "payload": "session_depth"},
    "order": {"params": "order_rate", "stream": "order", "payload": None},
    "support": {"params": "support_rate", "stream": "support", "payload": "sentiment"},
}
_CYCLE_FAMILIES = {
    "payment_fail": {"params": "payment_fail", "stream": "payment"},
    "downgrade": {"params": "downgrade", "stream": "downgrade"},
}
_PAYLOAD_STREAM = {"session_depth": "session_pages", "sentiment": "sentiment"}
_PAYLOAD_CLIP = {"session_depth": (1.0, np.inf), "sentiment": (-1.0, 1.0)}


def _coeffs(params: dict, key: str, null_stream: bool) -> dict:
    """Frozen (a, b[, tau]) for an event family; null-stream control zeroes the health slope b."""
    c = dict(params["events"][key])
    if null_stream:
        c["b"] = 0.0
    return c


def _origin_ns(t0: pd.Timestamp) -> int:
    return int(pd.Timestamp(t0).value) - 52 * _WEEK_NS


def generate_events(
    customer_ids: NDArray,
    health_paths: NDArray[np.float64],
    params: dict,
    streams: dict[str, np.random.Generator],
    t0: pd.Timestamp = ANCHOR_T0,
    null_stream: bool = False,
) -> pd.DataFrame:
    """The full event log for ``customer_ids`` given their weekly ``health_paths`` (n, W+1)."""
    customer_ids = np.asarray(customer_ids, dtype=object)
    health_paths = np.asarray(health_paths, dtype=np.float64)
    weeks = int(params["latent"]["weeks"])
    origin_ns = _origin_ns(t0)

    frames = [
        _rate_family(
            name, spec, customer_ids, health_paths, weeks, origin_ns, params, streams, null_stream
        )
        for name, spec in _RATE_FAMILIES.items()
    ]
    frames += [
        _cycle_family(
            name, spec, customer_ids, health_paths, weeks, origin_ns, params, streams, null_stream
        )
        for name, spec in _CYCLE_FAMILIES.items()
    ]
    out = pd.concat(frames, ignore_index=True)
    # Canonical, deterministic ordering (independent of family concat order) for reproducible bytes.
    out = out.sort_values(["customer_id", "event_ts", "event_type"], kind="stable").reset_index(
        drop=True
    )
    out["event_ts"] = out["event_ts"].astype("datetime64[ns]")
    return out[["event_id", "customer_id", "event_ts", "event_type", "value"]]


def _rate_family(
    name, spec, customer_ids, health, weeks, origin_ns, params, streams, null_stream
) -> pd.DataFrame:
    """Poisson-count-per-week realization of the frozen Exponential-arrival rate process (D6.2)."""
    rng = streams[spec["stream"]]
    n = health.shape[0]
    c = _coeffs(params, spec["params"], null_stream)
    lam = K.exp_link(health[:, :weeks], c["a"], c["b"])  # (n, W) weekly rate at start-of-week h_w
    counts = rng.poisson(lam)  # (n, W)
    total = int(counts.sum())
    if total == 0:
        return _empty_frame()

    flat = counts.ravel()  # index = i*W + w (row-major: weeks ascending per customer)
    pair = np.repeat(np.arange(n * weeks), flat)
    cust_of = pair // weeks
    week_of = pair % weeks

    within_week = rng.integers(0, _WEEK_NS, size=total)  # uniform placement inside the week
    ts_ns = origin_ns + week_of.astype(np.int64) * _WEEK_NS + within_week

    value = _payload(spec["payload"], health[cust_of, week_of], params, streams, null_stream)
    return _assemble(name, customer_ids, cust_of, ts_ns, value, counts.sum(axis=1))


def _cycle_family(
    name, spec, customer_ids, health, weeks, origin_ns, params, streams, null_stream
) -> pd.DataFrame:
    """Per-4-week-cycle Bernoulli with placement-week evaluation (D6.2 major fix)."""
    rng = streams[spec["stream"]]
    n = health.shape[0]
    c = _coeffs(params, spec["params"], null_stream)
    n_cycles = weeks // CYCLE_WEEKS

    cust_list, ts_list = [], []
    for cyc in range(n_cycles):
        # Draw the placement week first, evaluate the hazard THERE -> existence depends only on
        # health at/before the event's own timestamp (no terminal-week look-ahead).
        w_star = cyc * CYCLE_WEEKS + rng.integers(0, CYCLE_WEEKS, size=n)
        h_star = health[np.arange(n), w_star]
        p = K.logistic_link(h_star, c["a"], c["b"])
        hit = rng.random(n) < p
        if not hit.any():
            continue
        cust_of = np.nonzero(hit)[0]
        within = rng.integers(0, _WEEK_NS, size=cust_of.size)
        ts_ns = origin_ns + w_star[cust_of].astype(np.int64) * _WEEK_NS + within
        cust_list.append(cust_of)
        ts_list.append(ts_ns)

    if not cust_list:
        return _empty_frame()
    cust_of = np.concatenate(cust_list)
    ts_ns = np.concatenate(ts_list)
    # Per-customer event count (for stable within-(customer, type) sequence ids).
    counts = np.bincount(cust_of, minlength=n)
    order = np.argsort(cust_of, kind="stable")  # group by customer for contiguous seq assignment
    return _assemble(name, customer_ids, cust_of[order], ts_ns[order], None, counts)


def _payload(payload, h_at_event, params, streams, null_stream):
    if payload is None:
        return None
    c = _coeffs(params, payload, null_stream)
    rng = streams[_PAYLOAD_STREAM[payload]]
    mean = K.linear_link(h_at_event, c["a"], c["b"])
    draw = mean + c["tau"] * rng.standard_normal(h_at_event.shape[0])
    lo, hi = _PAYLOAD_CLIP[payload]
    return np.clip(draw, lo, hi)


def _assemble(name, customer_ids, cust_of, ts_ns, value, per_customer_counts) -> pd.DataFrame:
    """Build a family frame; event_id = '{customer_id}:{type}:{seq}' (stable idempotency key, D6.1).

    ``cust_of`` must be grouped by customer (contiguous) so the per-(customer, type) sequence is
    assigned by simple within-group position.
    """
    total = cust_of.shape[0]
    offsets = np.concatenate([[0], np.cumsum(per_customer_counts)])[:-1]
    seq = np.arange(total) - offsets[cust_of]
    ids = customer_ids[cust_of].astype(str)
    event_id = pd.Series(ids) + f":{name}:" + pd.Series(seq.astype(str))
    return pd.DataFrame(
        {
            "event_id": event_id.to_numpy(),
            "customer_id": ids,
            "event_ts": ts_ns.astype("datetime64[ns]"),
            "event_type": name,
            "value": np.full(total, np.nan) if value is None else np.asarray(value, dtype=float),
        }
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": pd.Series([], dtype=object),
            "customer_id": pd.Series([], dtype=object),
            "event_ts": pd.Series([], dtype="datetime64[ns]"),
            "event_type": pd.Series([], dtype=object),
            "value": pd.Series([], dtype=float),
        }
    )
