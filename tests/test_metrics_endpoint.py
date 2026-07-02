"""Prometheus /metrics endpoint: real instrumentation on a FastAPI app (no mocks)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.metrics import add_metrics  # noqa: E402


def test_metrics_endpoint_records_requests():
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    add_metrics(app, "test-svc")
    client = TestClient(app)

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200

    body = client.get("/metrics").text
    assert "churn_http_requests_total" in body
    assert "churn_http_request_duration_seconds" in body
    assert 'service="test-svc"' in body
    assert 'path="/ping"' in body
