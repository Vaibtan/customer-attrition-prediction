"""Observability wiring is configured: Prometheus scrapes the API; Grafana is provisioned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_grafana_dashboard_json_is_valid_and_uses_our_metrics():
    dash = json.loads((ROOT / "infra/grafana/dashboards/churn-ops.json").read_text())
    assert dash["uid"] == "churn-ops"
    exprs = [t["expr"] for p in dash["panels"] for t in p.get("targets", [])]
    assert any("churn_http_requests_total" in e for e in exprs)
    assert any("churn_http_request_duration_seconds_bucket" in e for e in exprs)
    assert any("churn_model_state" in e for e in exprs)  # the ADR 0006 serving-state panel


def _load_alerts() -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load((ROOT / "infra/prometheus/alerts.yml").read_text())


def test_alert_rules_are_well_formed():
    pytest.importorskip("yaml")
    rules = [r for g in _load_alerts()["groups"] for r in g["rules"]]
    names = {r["alert"] for r in rules}
    assert names == {
        "HighP99Latency",
        "High5xxRatio",
        "ScrapeTargetDown",
        "ModelStateDegradedOrStale",
    }
    for r in rules:
        assert r["expr"].strip()
        assert r["for"]  # every alert waits before firing (no flap on a single scrape)
        assert r["labels"]["severity"] in {"warning", "critical"}
        assert r["annotations"]["summary"]
        assert r["annotations"]["runbook"].startswith("docs/RUNBOOK.md#")


def test_model_state_alert_watches_the_gauge():
    """Item 4 <-> item 3: the platform's real failure-mode alert fires on churn_model_state going
    degraded|stale (ADR 0006), so the new gauge is actually wired to a page-worthy rule."""
    pytest.importorskip("yaml")
    rules = {r["alert"]: r for g in _load_alerts()["groups"] for r in g["rules"]}
    expr = rules["ModelStateDegradedOrStale"]["expr"]
    assert "churn_model_state" in expr
    assert "degraded" in expr and "stale" in expr


def test_prometheus_loads_the_rule_file_and_compose_mounts_it():
    prom = (ROOT / "infra/prometheus.yml").read_text()
    assert "rule_files:" in prom
    assert "/etc/prometheus/alerts.yml" in prom
    compose = (ROOT / "compose.yaml").read_text()
    assert "./infra/prometheus/alerts.yml:/etc/prometheus/alerts.yml:ro" in compose


def test_runbook_has_a_resolvable_entry_per_alert():
    """No doc drift (REV-17 lesson): every alert's runbook anchor resolves to a runbook header."""
    pytest.importorskip("yaml")
    runbook = (ROOT / "docs/RUNBOOK.md").read_text()
    # GitHub-style anchors: '## HighP99Latency' -> '#highp99latency'
    headers = {
        "#" + line[3:].strip().lower().replace(" ", "-")
        for line in runbook.splitlines()
        if line.startswith("## ")
    }
    for r in (r for g in _load_alerts()["groups"] for r in g["rules"]):
        anchor = "#" + r["annotations"]["runbook"].split("#", 1)[1]
        assert anchor in headers, f"{r['alert']} runbook anchor {anchor} has no section"


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
