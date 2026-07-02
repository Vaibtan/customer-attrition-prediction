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


def test_prometheus_scrapes_the_scoring_api():
    text = (ROOT / "infra/prometheus.yml").read_text()
    assert "churn-api" in text
    assert "api:8000" in text


def test_grafana_datasource_and_dashboard_provider_provisioned():
    ds = (ROOT / "infra/grafana/provisioning/datasources/prometheus.yml").read_text()
    assert "http://prometheus:9090" in ds
    assert "uid: prometheus" in ds
    provider = (ROOT / "infra/grafana/provisioning/dashboards/provider.yml").read_text()
    assert "/var/lib/grafana/dashboards" in provider
