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


def test_metrics_label_uses_route_template_not_raw_url():
    """Unmatched/scanner URLs share one bucket -- raw paths would mint unbounded timeseries."""
    app = FastAPI()

    @app.get("/items/{item_id}")
    def item(item_id: str):
        return {"item_id": item_id}

    add_metrics(app, "cardinality-svc")
    client = TestClient(app)

    assert client.get("/items/a").status_code == 200
    assert client.get("/items/b").status_code == 200
    assert client.get("/wp-admin/setup.php").status_code == 404

    body = client.get("/metrics").text
    assert 'path="/items/{item_id}"' in body  # template, one series for all item ids
    assert 'path="/items/a"' not in body
    assert 'path="<unmatched>"' in body  # 404s bucketed, not per-URL
    assert "wp-admin" not in body


def test_metrics_count_uncaught_exceptions_as_500():
    """An exception that escapes the route must still increment the counter (5xx panel input)."""
    app = FastAPI()

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    add_metrics(app, "error-svc")
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/boom").status_code == 500

    body = client.get("/metrics").text
    series = 'churn_http_requests_total{method="GET",path="/boom",service="error-svc",status="500"}'
    assert series in body
