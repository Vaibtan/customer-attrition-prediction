"""Shared FastAPI scoring-app scaffold (ADR 0005).

Both ``/score`` (``serve.py``) and ``/score/online`` (``online.py``) build on :func:`scoring_app`:
it owns app creation, a load-once model cache, the ``FileNotFoundError -> degraded / 503`` contract,
the ``/health`` payload, and the Prometheus metrics middleware. Each app supplies its request /
response models and a ``score_fn`` that owns everything bespoke -- the batch DataFrame build +
validate, the online Redis read + 404. The one shared contract is that ``loader()`` returns an
object exposing ``.run_id``.

NOTE: no ``from __future__ import annotations`` here on purpose -- the ``score`` route is annotated
with the *parameter* ``request_model``, so the annotation must stay a real class object for FastAPI
to parse the body (a stringized annotation would fail to resolve the local name).
"""

import threading
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.metrics import add_metrics


def scoring_app(
    *,
    title: str,
    description: str,
    path: str,
    loader: Callable[[], object],
    request_model: type[BaseModel],
    response_model: type[BaseModel],
    score_fn: Callable[[object, BaseModel], BaseModel],
    metrics_label: str,
) -> FastAPI:
    """Build a scoring FastAPI app around a model ``loader`` and a bespoke ``score_fn``.

    ``loader() -> model`` is called once (lazily) and cached; the model must expose ``.run_id``. A
    missing model (``FileNotFoundError``) degrades ``/health`` and 503s scoring. ``score_fn(model,
    request) -> response`` owns all per-endpoint logic and may raise its own HTTP errors (e.g. the
    online path's 404 on a Redis cache miss).
    """
    app = FastAPI(title=title, version="0.1.0", description=description)
    holder: dict[str, object] = {}
    load_lock = threading.Lock()

    def get_model() -> object:
        # Handlers are sync `def`s, so Starlette runs them in a threadpool: without the lock two
        # concurrent first requests would each run loader() (double model load).
        if "model" not in holder:
            with load_lock:
                if "model" not in holder:
                    holder["model"] = loader()
        return holder["model"]

    @app.get("/health")
    def health():
        try:
            model = get_model()
        except FileNotFoundError:
            return {"status": "degraded", "model_loaded": False}
        return {"status": "ok", "model_loaded": True, "run_id": model.run_id}

    @app.post(path, response_model=response_model)
    def score(payload: request_model):  # type: ignore[valid-type]
        try:
            model = get_model()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return score_fn(model, payload)

    add_metrics(app, metrics_label)
    return app
