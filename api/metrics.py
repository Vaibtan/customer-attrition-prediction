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
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
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
