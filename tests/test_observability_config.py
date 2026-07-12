"""Observability wiring is configured: Prometheus scrapes the API; Grafana is provisioned."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_grafana_dashboard_json_is_valid_and_uses_our_metrics():
    dash = json.loads((ROOT / "infra/grafana/dashboards/churn-ops.json").read_text())
    assert dash["uid"] == "churn-ops"
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    assert any("churn_http_requests_total" in e for e in exprs)
    assert any("churn_http_request_duration_seconds_bucket" in e for e in exprs)


def test_prometheus_scrapes_both_scoring_apis():
    text = (ROOT / "infra/prometheus.yml").read_text()
    assert "churn-api" in text
    assert "api:8000" in text
    # ADR 0006: the online app is a real compose service now -- its service label must be able
    # to exist (the old config scraped batch only, so `online-scoring` could never appear).
    assert "churn-api-online" in text
    assert "api-online:8001" in text


def test_mlflow_server_pin_matches_the_client_lock():
    """The server image pins mlflow outside uv.lock; this is the sync guard (REV-17)."""
    import re  # noqa: PLC0415

    dockerfile = (ROOT / "infra/Dockerfile.mlflow").read_text()
    pin = re.search(r"mlflow==([\d.]+)", dockerfile)
    assert pin is not None, "Dockerfile.mlflow must pin an exact mlflow version"
    lock = (ROOT / "uv.lock").read_text()
    locked = re.search(r'name = "mlflow"\nversion = "([\d.]+)"', lock)
    assert locked is not None
    assert pin.group(1) == locked.group(1), (
        f"server pin {pin.group(1)} != locked client {locked.group(1)}"
    )


def test_grafana_datasource_and_dashboard_provider_provisioned():
    ds = (ROOT / "infra/grafana/provisioning/datasources/prometheus.yml").read_text()
    assert "http://prometheus:9090" in ds
    assert "uid: prometheus" in ds
    provider = (ROOT / "infra/grafana/provisioning/dashboards/provider.yml").read_text()
    assert "/var/lib/grafana/dashboards" in provider
