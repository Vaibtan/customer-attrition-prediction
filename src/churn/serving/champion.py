"""Champion resolution for online serving: promotion really moves production (ADR 0006).

The online app follows the MLflow ``@champion`` alias instead of a pinned artifact. The resolver
re-checks the alias VERSION lazily on the request path once a TTL has elapsed (cheap registry
call), and reloads the artifact only when the version actually changed -- no background thread,
single-worker friendly, ``ttl_seconds=0`` makes cutover deterministic in tests.

Outage policy -- *serve-last-good*: a failed refresh with a model already loaded keeps serving it
and surfaces as ``status: stale`` in :meth:`ChampionResolver.health_status` (stale still scores;
CONTEXT.md). Only "no model has EVER been resolved" raises :class:`ModelUnavailable`, which the
scoring scaffold maps to 503 / a degraded health body. A failed refresh is retried on the next
request past the TTL (no backoff -- the version check is one cheap registry call).

MLflow imports stay inside :func:`mlflow_alias_resolver` so this module (and the resolver's unit
tests) never require the ``tracking`` extra.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping

from churn.serving.online_model import OnlineModel, OnlineScorer

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


class ModelUnavailable(RuntimeError):
    """No model has ever been resolved -- scoring must refuse, never guess."""


class ChampionResolver:
    """Lazy-TTL alias follower. Duck-types the scorer contract (``score`` + ``run_id``)."""

    def __init__(
        self,
        version_fn: Callable[[], str],
        load_fn: Callable[[str], OnlineModel],
        ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        self._version_fn = version_fn
        self._load_fn = load_fn
        self.ttl = float(ttl_seconds)
        self._clock = clock
        # on_state fires on each state TRANSITION (ok|degraded|stale) from inside a resolve -- the
        # API layer wires it to a Prometheus gauge (churn_model_state). Same callback seam as the
        # shadow on_outcome counter: prometheus_client never becomes a dependency of churn.serving.
        self._on_state = on_state or (lambda state: None)
        self._lock = threading.Lock()
        self._version: str | None = None
        self._model: OnlineModel | None = None
        self._resolved_at: float = float("-inf")
        self._refresh_error: str | None = None
        self._state: str | None = None

    def _set_state(self, state: str) -> None:
        """Emit a model-source state transition to on_state, but only on an ACTUAL change. Any
        _maybe_refresh caller (score OR health_status) can reach here, but only when a resolve
        actually runs: within-TTL calls early-return first, so /health poll frequency never churns
        the gauge -- it tracks what the resolver did, not how often it was polled."""
        if state != self._state:
            self._state = state
            self._on_state(state)

    def _maybe_refresh(self) -> None:
        with self._lock:
            now = self._clock()
            if self._model is not None and (now - self._resolved_at) < self.ttl:
                return
            try:
                version = self._version_fn()
                if version != self._version:
                    self._model = self._load_fn(version)  # artifact load only on a version move
                    self._version = version
                self._resolved_at = now
                self._refresh_error = None
                self._set_state("ok")
            except Exception as exc:
                self._refresh_error = f"{type(exc).__name__}: {exc}"
                if self._model is None:
                    self._set_state("degraded")
                    raise ModelUnavailable(f"champion unavailable: {self._refresh_error}") from exc
                # serve-last-good: keep the loaded model, report stale via health_status.
                self._set_state("stale")

    @property
    def run_id(self) -> str | None:
        self._maybe_refresh()
        return self._model.run_id

    def score(self, static: Mapping[str, object], event: Mapping[str, object]):
        self._maybe_refresh()
        return OnlineScorer(self._model).score(static, event)

    def health_status(self) -> dict:
        """The /health body (never raises): ok | stale | degraded, with provenance."""
        try:
            self._maybe_refresh()
        except ModelUnavailable:
            return {"status": "degraded", "model_loaded": False, "reason": self._refresh_error}
        body = {
            "status": "stale" if self._refresh_error else "ok",
            "model_loaded": True,
            "run_id": self._model.run_id,
            "alias_version": self._version,
            "resolved_age_seconds": round(self._clock() - self._resolved_at, 3),
        }
        if self._refresh_error:
            body["reason"] = self._refresh_error
        return body


def mlflow_alias_resolver(
    name: str,
    alias: str,
    tracking_uri: str | None,
    ttl_seconds: float = 60.0,
    on_state: Callable[[str], None] | None = None,
) -> ChampionResolver:
    """A resolver bound to ``models:/{name}@{alias}`` on an MLflow registry.

    ``tier_cutpoints`` ride on the model VERSION as a JSON tag (set by
    ``mlflow_registry.log_and_register``). An unset ``tracking_uri`` yields a resolver that is
    permanently degraded (never a silent fallback to a local ./mlruns file store). ``on_state``
    (optional) receives ok|degraded|stale transitions -- the API wires the champion resolver's to
    the churn_model_state gauge (the challenger/shadow resolver leaves it unset).
    """

    def version_fn() -> str:
        if not tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI unset -- cannot resolve the champion alias")
        from churn.lifecycle import mlflow_registry as MR  # tracking extra, imported lazily

        return MR.alias_version(name, alias, tracking_uri)

    def load_fn(version: str) -> OnlineModel:
        import mlflow.sklearn  # noqa: PLC0415
        from mlflow import MlflowClient  # noqa: PLC0415

        client = MlflowClient(tracking_uri=tracking_uri)
        tags = dict(client.get_model_version(name, version).tags or {})
        if "tier_cutpoints" not in tags:
            raise RuntimeError(
                f"model version {name} v{version} carries no tier_cutpoints tag -- "
                "register it via mlflow_registry.log_and_register(..., tags=...)"
            )
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(tracking_uri)
        pipeline = mlflow.sklearn.load_model(f"models:/{name}/{version}")
        return OnlineModel(
            pipeline=pipeline,
            cutpoints=json.loads(tags["tier_cutpoints"]),
            run_id=f"{name}@v{version}",
        )

    return ChampionResolver(version_fn, load_fn, ttl_seconds=ttl_seconds, on_state=on_state)
