"""Prometheus instrumentation for the FastAPI apps (ops-only): request count, latency, errors.

Adds a ``/metrics`` endpoint (Prometheus text format) + a middleware that times every request.
Prometheus scrapes it and Grafana visualises it (compose ``observability`` profile). These are
**operational** signals only -- request latency/throughput/error rate. ML + drift signals live in
the Streamlit mission-control, kept separate on purpose (ENHANCEMENT_PLAN.md Sec 7).

Metrics are module-level (registered once on the default registry) so instrumenting multiple apps
never double-registers a timeseries; the ``service`` label distinguishes them.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

_REQUESTS = Counter(
    "churn_http_requests_total",
    "Total HTTP requests.",
    ["service", "method", "path", "status"],
)
_LATENCY = Histogram(
    "churn_http_request_duration_seconds",
    "HTTP request latency (seconds).",
    ["service", "method", "path"],
)
# Serving model-source state as a one-hot gauge (ADR 0006's actual failure mode: the alias-
# following online app going stale/degraded). Set from inside the ChampionResolver transitions
# via its on_state callback -- NOT from /health polling. Only the online app has a resolver, but
# the `service` label keeps this app-agnostic like the request metrics above.
_MODEL_STATES = ("ok", "degraded", "stale")
_MODEL_STATE = Gauge(
    "churn_model_state",
    "Serving model-source state, one-hot per service (1 = current state).",
    ["service", "state"],
)


def set_model_state(service: str, state: str) -> None:
    """Light ``state`` for ``service`` and clear the others -- so the degraded/stale alert can
    never match a stale leftover series. ``state`` must be one of :data:`_MODEL_STATES`."""
    if state not in _MODEL_STATES:
        raise ValueError(f"unknown model state {state!r}; expected one of {_MODEL_STATES}")
    for s in _MODEL_STATES:
        _MODEL_STATE.labels(service, s).set(1.0 if s == state else 0.0)


def add_metrics(app: FastAPI, service: str) -> FastAPI:
    """Wire request timing + a ``/metrics`` endpoint onto ``app`` under the given service label."""

    @app.middleware("http")
    async def _instrument(request: Request, call_next):
        # Label with the matched route TEMPLATE, not the raw URL: raw paths mint a new timeseries
        # per scanner/typo URL (unbounded cardinality). Routing runs inside call_next, so the
        # matched route is only on the scope afterwards; unmatched requests share one bucket.
        # The try/finally guarantees uncaught 500s are counted too -- call_next re-raises before
        # any code after it runs, which is exactly how the 5xx panel goes blind.
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = route.path if route is not None else "<unmatched>"
            if path != "/metrics":  # don't measure the scrape itself
                _LATENCY.labels(service, request.method, path).observe(elapsed)
                _REQUESTS.labels(service, request.method, path, str(status)).inc()
        return response

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
