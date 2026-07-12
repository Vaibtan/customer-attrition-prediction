# infra/

Deployment infrastructure, referenced by `compose.yaml` at the repo root:

- `Dockerfile.streaming` — producer + Quix consumer image (streaming profile).
- `Dockerfile.online` — the alias-following online scoring API (`api-online:8001`, ADR 0006); no
  baked model on purpose — it resolves the MLflow `@champion`.
- `Dockerfile.mlflow` — MLflow tracking + model-registry server (sqlite backend, proxied
  artifacts). The `mlflow==` pin must match `uv.lock`'s client version — guarded by
  `tests/test_observability_config.py`.
- `Dockerfile.dashboard` — Streamlit mission-control image.
- `Dockerfile.test` — the real-infra integration test-runner (ADR 0002); the repo is bind-mounted
  read-only at run time.
- `prometheus.yml` — ops-only scrape config for the FastAPI `/metrics` endpoints. NOTE: the
  `observability` profile scrapes services from the `serve` profile — run both for live panels.
- `grafana/` — provisioned datasource + the churn-ops dashboard JSON.
