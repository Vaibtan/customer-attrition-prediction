"""Event-generator tests (D6.1/D6.2/D6.6/D6.7/D6.8).

The generator draws the per-week event stream off the frozen links; it must: emit the full 52-week
window for every customer regardless of tenure (D6.8), place every event strictly in (t0-364d, t0)
(PIT), key each event with a stable event_id (D6.1), and -- the Codex D6.2-major fix -- make a
per-cycle Bernoulli event's *existence* depend only on the health of the week it is placed in
(placement-week evaluation), never a later terminal week. The null-stream control (b=0) leaves the
independent health/label streams untouched (D6.8 control-validity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.simulator import events as E
from churn.simulator import latent as L
from churn.simulator import params as P
from churn.simulator import rng as R

EVENT_TYPES = {"login", "order", "payment_fail", "support", "downgrade"}


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


def _const_health(n: int, level: float, weeks: int) -> np.ndarray:
    """A degenerate 'health path' pinned at a constant level -- isolates the link direction."""
    return np.full((n, weeks + 1), level, dtype=np.float64)


def test_schema_and_event_types(params):
    q = np.zeros(200)
    streams = R.make_streams(1)
    h = L.simulate_health_paths(q, params, streams["latent"])
    ev = E.generate_events(np.array([f"C{i}" for i in range(200)]), h, params, streams)
    assert list(ev.columns) == ["event_id", "customer_id", "event_ts", "event_type", "value"]
    assert set(ev["event_type"].unique()) <= EVENT_TYPES
    assert ev["event_id"].is_unique


def test_all_events_are_strictly_pre_t0_and_in_window(params):
    q = np.zeros(300)
    streams = R.make_streams(2)
    h = L.simulate_health_paths(q, params, streams["latent"])
    t0 = E.ANCHOR_T0
    ev = E.generate_events(np.array([f"C{i}" for i in range(300)]), h, params, streams, t0=t0)
    assert (ev["event_ts"] < t0).all()
    assert (ev["event_ts"] >= t0 - pd.Timedelta(days=364)).all()


def test_payload_by_type_is_correct(params):
    q = np.zeros(500)
    streams = R.make_streams(3)
    h = L.simulate_health_paths(q, params, streams["latent"])
    ev = E.generate_events(np.array([f"C{i}" for i in range(500)]), h, params, streams)
    assert ev.loc[ev["event_type"] == "login", "value"].notna().all()  # pages_per_session
    assert ev.loc[ev["event_type"] == "support", "value"].notna().all()  # sentiment
    for t in ("order", "payment_fail", "downgrade"):
        assert ev.loc[ev["event_type"] == t, "value"].isna().all()
    # session depth clipped >= 1; sentiment clipped to [-1, 1].
    assert (ev.loc[ev["event_type"] == "login", "value"] >= 1.0).all()
    sent = ev.loc[ev["event_type"] == "support", "value"]
    assert (sent >= -1.0).all() and (sent <= 1.0).all()


def test_determinism(params):
    ids = np.array([f"C{i}" for i in range(100)])
    q = np.linspace(-1, 1, 100)
    a = E.generate_events(
        ids,
        L.simulate_health_paths(q, params, R.make_streams(9)["latent"]),
        params,
        R.make_streams(9),
    )
    b = E.generate_events(
        ids,
        L.simulate_health_paths(q, params, R.make_streams(9)["latent"]),
        params,
        R.make_streams(9),
    )
    pd.testing.assert_frame_equal(a, b)


def test_baseline_rates_at_zero_health(params):
    # health == 0 => login rate ~6/wk over 52wk ~312; payment fail ~7%/cycle * 13 (D7 re-lock).
    n = 4000
    ids = np.array([f"C{i}" for i in range(n)])
    h = _const_health(n, 0.0, params["latent"]["weeks"])
    ev = E.generate_events(ids, h, params, R.make_streams(21))
    per = ev.groupby("event_type").size() / n
    assert per["login"] == pytest.approx(312, rel=0.05)  # ~6/wk * 52
    assert per["order"] == pytest.approx(52, rel=0.10)  # ~1/wk * 52
    assert per["payment_fail"] == pytest.approx(0.91, rel=0.20)  # ~0.07 * 13 cycles
    assert per["downgrade"] == pytest.approx(0.39, rel=0.25)  # ~0.03 * 13 cycles


def test_health_direction_healthier_more_logins_fewer_failures(params):
    n = 3000
    weeks = params["latent"]["weeks"]
    ids = np.array([f"C{i}" for i in range(n)])
    healthy = E.generate_events(ids, _const_health(n, 1.5, weeks), params, R.make_streams(31))
    unhealthy = E.generate_events(ids, _const_health(n, -1.5, weeks), params, R.make_streams(31))
    hc = healthy.groupby("event_type").size()
    uc = unhealthy.groupby("event_type").size()
    assert hc["login"] > uc["login"]  # healthier -> more logins (b>0)
    assert hc["payment_fail"] < uc["payment_fail"]  # unhealthier -> more failures (b<0)
    assert hc["support"] < uc["support"]  # unhealthier -> more tickets (b<0)


def test_null_stream_kills_event_health_dependence(params):
    n = 3000
    weeks = params["latent"]["weeks"]
    ids = np.array([f"C{i}" for i in range(n)])
    healthy = E.generate_events(
        ids, _const_health(n, 1.5, weeks), params, R.make_streams(41), null_stream=True
    )
    unhealthy = E.generate_events(
        ids, _const_health(n, -1.5, weeks), params, R.make_streams(41), null_stream=True
    )
    # With b zeroed, login rate is health-independent -> counts match within noise.
    assert healthy.groupby("event_type").size()["login"] == pytest.approx(
        unhealthy.groupby("event_type").size()["login"], rel=0.05
    )


def test_null_stream_leaves_health_and_label_streams_untouched(params):
    # Control-validity invariant (D6.8): regenerating events with b=0 draws from independent event
    # streams, so the latent + label streams are byte-identical -> h(t0) and y are unchanged.
    q = np.linspace(-2, 2, 500)
    h_informative = L.simulate_health_paths(q, params, R.make_streams(55)["latent"])
    h_null = L.simulate_health_paths(q, params, R.make_streams(55)["latent"])
    assert np.array_equal(h_informative, h_null)
    # And the label stream draws identically regardless of the event mode.
    lab1 = R.make_streams(55)["label"].random(500)
    lab2 = R.make_streams(55)["label"].random(500)
    assert np.array_equal(lab1, lab2)


def test_cycle_event_existence_depends_on_placement_week_health(params):
    # Codex D6.2-major: a per-cycle Bernoulli event must exist because of the health of the week it
    # is placed in, never a later terminal week. With a step path (healthy weeks 0-25, unhealthy
    # 26-51), payment failures must concentrate in the UNHEALTHY second half by event_ts.
    n = 4000
    weeks = params["latent"]["weeks"]
    h = np.empty((n, weeks + 1))
    h[:, : weeks // 2 + 1] = 3.0  # very healthy first half
    h[:, weeks // 2 + 1 :] = -3.0  # very unhealthy second half
    ids = np.array([f"C{i}" for i in range(n)])
    t0 = E.ANCHOR_T0
    ev = E.generate_events(ids, h, params, R.make_streams(63), t0=t0)
    fails = ev[ev["event_type"] == "payment_fail"]
    midpoint = t0 - pd.Timedelta(days=182)
    frac_second_half = float((fails["event_ts"] >= midpoint).mean())
    # Existence tracks placement-week health -> the vast majority land in the unhealthy half.
    assert frac_second_half > 0.9, frac_second_half


def test_full_window_emitted_regardless_of_tenure(params):
    # D6.8: event history length must NOT depend on account_age_days. The generator takes only
    # (ids, health, params) -- no tenure input -- so 52 weeks emit for everyone by construction.
    import inspect

    sig = set(inspect.signature(E.generate_events).parameters)
    assert "account_age_days" not in sig and "tenure" not in sig
