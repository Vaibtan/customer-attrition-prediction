# Runbook — churn scoring platform alerts

Operational responses for the Prometheus alerts in [`infra/prometheus/alerts.yml`](../infra/prometheus/alerts.yml).

## How alerting terminates here

In production these alerts would route through **Alertmanager → a pager** (PagerDuty/Opsgenie/Slack).
This demo deliberately ships **no Alertmanager** — a receiver-less Alertmanager is theatre (a locked
decision, `REVIEW_ISSUES.md` GRILLED 2026-07-16). Alerts terminate at:

- the **Prometheus `/alerts` UI** — <http://localhost:9090/alerts> (state: inactive / pending / firing),
  and `/api/v1/rules` for the loaded rule set;
- the Grafana **"Model source state"** panel on the *Churn Scoring API — Ops* dashboard
  (<http://localhost:3000>, `observability` profile).

Bring the relevant stack up with (a local `.env` with `GRAFANA_ADMIN_PASSWORD` is required — see
`.env.example`):

```bash
docker compose --profile serve --profile observability up -d
```

Each alert carries a `runbook` annotation pointing at the matching section below. The compose
commands assume you run them from the repo root.

---

## HighP99Latency

**Means:** 5-minute p99 request latency on a scoring service is above 500ms. A single sklearn predict
(batch) or a Redis read + predict (online) is milliseconds, so a sustained p99 this high means the
scorer or a dependency (Redis, the registry/tracking resolve) is struggling, or the container is
starved.

**First checks:**
```bash
docker compose --profile serve --profile observability ps      # are api / api-online up (not restarting)?
docker compose logs --tail=100 api-online                       # slow-path errors, resolver retries?
docker stats --no-stream                                        # is the container pinned at its mem_limit / CPU?
```

**Recovery:** latency returns below threshold once the bottleneck clears (e.g. Redis healthy again,
or the container given headroom). The alert auto-resolves after the next scrape below 500ms.

---

## High5xxRatio

**Means:** more than 5% of requests to a service returned 5xx over 5 minutes. 5xx counting is truthful
(REV-12: uncaught exceptions increment the counter before re-raising), so this is real server-side
failure — most often a degraded model source making `/score` refuse, or an unhandled exception.

**First checks:**
```bash
docker compose logs --tail=200 api-online                       # the actual exceptions / tracebacks
curl -fsS http://localhost:8001/health                          # is the model degraded/stale? (degraded => /score 503s)
docker compose ps                                               # is the service crash-looping (restart count)?
```

**Recovery:** fix the underlying cause (see *ModelStateDegradedOrStale* if `/health` is not `ok`).
The ratio decays back under 5% as healthy requests replace the failing ones.

---

## ScrapeTargetDown

**Means:** Prometheus has been unable to scrape a scoring API target (`churn-api` or
`churn-api-online`) for 2 minutes — the process is down, crashed, or never came up.

**First checks:**
```bash
docker compose --profile serve ps                               # is the target container running?
docker compose logs --tail=100 api-online                       # did it fail on startup (bad env, import error)?
curl -fsS http://localhost:8001/metrics | head                  # is /metrics reachable from the host at all?
```

**Recovery:** `docker compose --profile serve up -d api-online` (or `api`). `up{job=~"churn-api.*"}`
returns to 1 within one scrape interval once the target answers `/metrics`.

---

## ModelStateDegradedOrStale

**Means:** the alias-following online app (ADR 0006) cannot resolve the `@champion` — the platform's
actual failure mode. `churn_model_state{state="degraded"}` = **no model ever loaded**, scoring 503s;
`state="stale"` = serving the **last-good** model while re-resolution keeps failing (stale still
scores). MLflow is a *soft* dependency: a stale state usually means tracking is down, not the app.

**First checks:**
```bash
curl -fsS http://localhost:8001/health                          # status + reason, alias_version, resolved_age_seconds
docker compose --profile tracking ps                            # is mlflow up? (stale => champion can't be re-resolved)
docker compose logs --tail=100 mlflow                           # why the resolve fails (server down, no @champion alias)
```

**Recovery:** **stale** self-heals — the resolver retries past its TTL and cuts over the moment
tracking returns (no restart needed). **degraded** means no champion has ever loaded: bring MLflow up
and ensure a version is aliased `@champion` (`churn.lifecycle.mlflow_registry.set_alias`). The gauge
flips back to `ok` on the next successful resolve.
