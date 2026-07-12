"""ChampionResolver (ADR 0006): lazy-TTL alias following, serve-last-good, honest health."""

from __future__ import annotations

import json

import pytest

from churn.serving.champion import ChampionResolver, ModelUnavailable
from churn.serving.online_model import OnlineModel
from churn.serving.shadow import ShadowScorer


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _model(version: str) -> OnlineModel:
    return OnlineModel(pipeline=None, cutpoints={}, run_id=f"m@v{version}")


def _make(ttl: float = 60.0):
    clock = _Clock()
    calls = {"version": 0, "load": 0}
    state = {"version": "1", "fail": False}

    def version_fn() -> str:
        calls["version"] += 1
        if state["fail"]:
            raise ConnectionError("tracking server down")
        return state["version"]

    def load_fn(version: str) -> OnlineModel:
        calls["load"] += 1
        return _model(version)

    resolver = ChampionResolver(version_fn, load_fn, ttl_seconds=ttl, clock=clock)
    return resolver, clock, calls, state


def test_ttl_gates_the_version_check_and_reload_only_on_version_move():
    resolver, clock, calls, state = _make(ttl=60.0)

    assert resolver.run_id == "m@v1"
    assert calls == {"version": 1, "load": 1}

    clock.now = 30.0  # inside the TTL: no registry traffic at all
    assert resolver.run_id == "m@v1"
    assert calls == {"version": 1, "load": 1}

    clock.now = 61.0  # past the TTL, version unchanged: re-check but NO artifact reload
    assert resolver.run_id == "m@v1"
    assert calls == {"version": 2, "load": 1}

    state["version"] = "2"
    clock.now = 122.0  # past the TTL, version moved: cutover
    assert resolver.run_id == "m@v2"
    assert calls == {"version": 3, "load": 2}


def test_never_resolved_is_degraded_and_refuses():
    resolver, _clock, _calls, state = _make()
    state["fail"] = True

    with pytest.raises(ModelUnavailable):
        _ = resolver.run_id
    body = resolver.health_status()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "tracking server down" in body["reason"]


def test_serve_last_good_goes_stale_then_recovers():
    resolver, clock, _calls, state = _make(ttl=60.0)
    assert resolver.run_id == "m@v1"  # loaded once, healthy

    state["fail"] = True
    clock.now = 61.0
    assert resolver.run_id == "m@v1"  # outage: still serving the last-good model
    body = resolver.health_status()
    assert body["status"] == "stale"
    assert body["model_loaded"] is True
    assert body["resolved_age_seconds"] == pytest.approx(61.0)
    assert "tracking server down" in body["reason"]

    state["fail"] = False
    state["version"] = "3"
    clock.now = 122.0
    assert resolver.run_id == "m@v3"  # recovery picks up the moved alias
    assert resolver.health_status()["status"] == "ok"


class _Score:
    churn_probability = 0.42


class _FakeChallenger:
    run_id = "challenger@v9"

    def score(self, static, event):
        return _Score()


class _NoChallenger:
    def score(self, static, event):
        raise ModelUnavailable("no @challenger aliased")


class _Broken:
    def score(self, static, event):
        raise RuntimeError("kaboom")


def test_shadow_logs_a_paired_record(tmp_path):
    outcomes: list[str] = []
    log = tmp_path / "shadow.jsonl"
    shadow = ShadowScorer(_FakeChallenger(), log, on_outcome=outcomes.append)

    ok = shadow.score_and_log(
        customer_id="C1",
        as_of="2025-01-01T00:00:00",
        static={"region": "North"},
        event={"login_count_90d": 3.0},
        champion_run_id="champ@v1",
        champion_probability=0.61,
    )
    assert ok is True and outcomes == ["scored"]
    record = json.loads(log.read_text().splitlines()[0])
    assert record["customer_id"] == "C1"
    assert record["champion_run_id"] == "champ@v1"
    assert record["champion_probability"] == 0.61
    assert record["challenger_run_id"] == "challenger@v9"
    assert record["challenger_probability"] == 0.42


def test_shadow_is_a_noop_without_a_challenger_and_never_raises(tmp_path):
    outcomes: list[str] = []
    log = tmp_path / "shadow.jsonl"
    kwargs = dict(
        customer_id="C1",
        as_of="t",
        static={},
        event={},
        champion_run_id=None,
        champion_probability=0.5,
    )

    silent = ShadowScorer(_NoChallenger(), log, on_outcome=outcomes.append)
    broken = ShadowScorer(_Broken(), log, on_outcome=outcomes.append)
    assert silent.score_and_log(**kwargs) is False
    assert broken.score_and_log(**kwargs) is False
    assert outcomes == ["no_challenger", "error"]
    assert not log.exists()  # nothing written on either failure path
