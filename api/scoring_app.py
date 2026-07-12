"""Shared FastAPI scoring-app scaffold (ADR 0005, health contract per ADR 0006).

Both ``/score`` (``serve.py``) and ``/score/online`` (``online.py``) build on :func:`scoring_app`:
it owns app creation, a locked load-once model cache, the degraded/503 contract, the ``/health``
payload, and the Prometheus metrics middleware. Each app supplies its request / response models
and a ``score_fn(model, request, background_tasks)`` that owns everything bespoke -- the batch
DataFrame build + validate, the online Redis read + 404/409 + shadow task. The shared loader
contract: ``loader()`` returns an object exposing ``.run_id``; it MAY also expose
``health_status() -> dict`` (a self-describing model source, e.g. the champion resolver reporting
ok|stale|degraded) and/or ``.warnings`` (surfaced in the health body).

Health contract (ADR 0006): one ``/health``, always HTTP 200, the body carries the state --
``ok | degraded | stale`` -- with a ``reason`` on anything but ok. ANY loader failure degrades
(not just a missing file: a corrupt artifact or missing metadata key is a routine MLOps failure
and must be reported, not 500'd). Scoring 503s only when no model is available.

NOTE: no ``from __future__ import annotations`` here on purpose -- the ``score`` route is annotated
with the *parameter* ``request_model``, so the annotation must stay a real class object for FastAPI
to parse the body (a stringized annotation would fail to resolve the local name).
"""

import threading
from collections.abc import Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from api.metrics import add_metrics
from churn.serving.champion import ModelUnavailable


def scoring_app(
    *,
    title: str,
    description: str,
    path: str,
    loader: Callable[[], object],
    request_model: type[BaseModel],
    response_model: type[BaseModel],
    score_fn: Callable[[object, BaseModel, BackgroundTasks], BaseModel],
    metrics_label: str,
) -> FastAPI:
    """Build a scoring FastAPI app around a model ``loader`` and a bespoke ``score_fn``."""
    app = FastAPI(title=title, version="0.1.0", description=description)
    holder: dict[str, object] = {}
    load_lock = threading.Lock()

    def get_model() -> object:
        # Handlers are sync `def`s, so Starlette runs them in a threadpool: without the lock two
        # concurrent first requests would each run loader() (double model load). Only success is
        # cached -- a failed load is retried on the next request.
        if "model" not in holder:
            with load_lock:
                if "model" not in holder:
                    holder["model"] = loader()
        return holder["model"]

    @app.get("/health")
    def health():
        try:
            model = get_model()
            status_fn = getattr(model, "health_status", None)
            if callable(status_fn):
                return status_fn()  # a self-describing source owns its ok|stale|degraded body
            body = {"status": "ok", "model_loaded": True, "run_id": model.run_id}
            warnings = list(getattr(model, "warnings", ()) or ())
            if warnings:
                body["warnings"] = warnings
            return body
        except Exception as exc:  # noqa: BLE001 -- ANY load failure degrades, with the reason
            return {
                "status": "degraded",
                "model_loaded": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    @app.post(path, response_model=response_model)
    def score(payload: request_model, background_tasks: BackgroundTasks):  # type: ignore[valid-type]
        try:
            model = get_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            return score_fn(model, payload, background_tasks)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    add_metrics(app, metrics_label)
    return app
