"""Shared scoring-app scaffold (ADR 0005): load-once cache semantics."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api.scoring_app import scoring_app  # noqa: E402


class _Req(BaseModel):
    x: float


class _Resp(BaseModel):
    y: float


class _Model:
    run_id = "run-1"


def test_concurrent_first_requests_load_the_model_once():
    """Two overlapping first requests must not each run loader() (unlocked check-then-act)."""
    calls: list[int] = []

    def slow_loader() -> _Model:
        calls.append(1)
        time.sleep(0.2)  # guarantee the second request arrives mid-load
        return _Model()

    app = scoring_app(
        title="t",
        description="d",
        path="/score",
        loader=slow_loader,
        request_model=_Req,
        response_model=_Resp,
        score_fn=lambda model, req: _Resp(y=req.x),
        metrics_label="lock-svc",
    )
    client = TestClient(app)

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: client.get("/health").status_code, range(2)))

    assert statuses == [200, 200]
    assert len(calls) == 1
