"""Shared scoring-app scaffold (ADR 0005): cache semantics + the ADR 0006 health contract."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api.scoring_app import scoring_app  # noqa: E402
from churn.serving.champion import ModelUnavailable  # noqa: E402


class _Req(BaseModel):
    x: float


class _Resp(BaseModel):
    y: float


class _Model:
    run_id = "run-1"


def _app(loader, score_fn=None):
    return scoring_app(
        title="t",
        description="d",
        path="/score",
        loader=loader,
        request_model=_Req,
        response_model=_Resp,
        score_fn=score_fn or (lambda model, req, tasks: _Resp(y=req.x)),
        metrics_label="scaffold-test",
    )


def test_concurrent_first_requests_load_the_model_once():
    """Two overlapping first requests must not each run loader() (unlocked check-then-act)."""
    calls: list[int] = []

    def slow_loader() -> _Model:
        calls.append(1)
        time.sleep(0.2)  # guarantee the second request arrives mid-load
        return _Model()

    client = TestClient(_app(slow_loader))

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: client.get("/health").status_code, range(2)))

    assert statuses == [200, 200]
    assert len(calls) == 1


def test_any_loader_failure_degrades_with_the_reason_not_a_500():
    """A corrupt artifact / missing metadata key is a routine failure -- report it (REV-10)."""

    def bad_loader():
        raise KeyError("tier_cutpoints")

    client = TestClient(_app(bad_loader))

    health = client.get("/health")
    assert health.status_code == 200  # the body carries the state, not the status code
    body = health.json()
    assert body == {
        "status": "degraded",
        "model_loaded": False,
        "reason": "KeyError: 'tier_cutpoints'",
    }
    assert client.post("/score", json={"x": 1.0}).status_code == 503


def test_failed_load_is_retried_and_recovers():
    attempts = {"n": 0}

    def flaky_loader() -> _Model:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FileNotFoundError("no run yet")
        return _Model()

    client = TestClient(_app(flaky_loader))
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/health").json()["status"] == "ok"  # only success is cached


def test_self_describing_source_owns_the_health_body():
    """A loader product with health_status() (the champion resolver) reports its own state."""

    class _Stale:
        run_id = "m@v1"

        def health_status(self) -> dict:
            return {
                "status": "stale",
                "model_loaded": True,
                "run_id": self.run_id,
                "alias_version": "1",
                "resolved_age_seconds": 120.0,
                "reason": "ConnectionError: tracking down",
            }

    body = TestClient(_app(lambda: _Stale())).get("/health").json()
    assert body["status"] == "stale"
    assert body["alias_version"] == "1"


def test_model_warnings_surface_in_health():
    class _Skewed(_Model):
        warnings = ("scikit_learn version skew: artifact 1.4.0, runtime 1.5.2",)

    body = TestClient(_app(lambda: _Skewed())).get("/health").json()
    assert body["status"] == "ok"
    assert "version skew" in body["warnings"][0]


def test_model_unavailable_from_score_fn_maps_to_503():
    """The resolver raises lazily inside score_fn; that is a 503, not an opaque 500."""

    def refuse(model, req, tasks):
        raise ModelUnavailable("champion unavailable: tracking down")

    client = TestClient(_app(lambda: _Model(), score_fn=refuse))
    resp = client.post("/score", json={"x": 1.0})
    assert resp.status_code == 503
    assert "champion unavailable" in resp.json()["detail"]
