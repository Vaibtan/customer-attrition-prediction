"""On-demand online scoring API: ``POST /score/online`` reads Redis features + request statics.

The scoped online path (ENHANCEMENT_PLAN.md Sec 3): a CSM/UI pulls up one customer; the endpoint
reads the latest event-feature envelope from Redis (written by the streaming consumer, keyed by
``customer_id``) and combines it with the static attributes in the request, scoring with the
deployed static+event model. Online features == offline PIT features (parity), so this equals the
offline batch score for the same ``(customer, t0)`` -- train/serve consistency at the score level.

Model source (ADR 0006): by default the app FOLLOWS the MLflow ``@champion`` alias via a lazy-TTL
:class:`~churn.serving.champion.ChampionResolver` -- promotion really moves production, MLflow is
a soft dependency (degraded until reachable, serve-last-good after), and, when an ``@challenger``
alias exists, every scored request is also shadow-scored post-response (log-only). An explicit
``run_dir`` (param or ``CHURN_ONLINE_MODEL_RUN_DIR``) pins a file-registry artifact instead --
the dev/test escape hatch and the parity-suite path.

Built on the shared :func:`api.scoring_app.scoring_app` scaffold (ADR 0005): the scaffold owns
the locked model cache, the degraded/stale/503 contract, and metrics; ``_score_online`` owns the
online-specific work (envelope read + 404/409 strict contract, static merge, the shadow task).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import Counter
from pydantic import BaseModel, Field

from api.scoring_app import scoring_app
from churn.featurestore.online import FeatureContractError, OnlineStore, validate_envelope
from churn.instrument import model as M
from churn.serving.champion import CHALLENGER_ALIAS, CHAMPION_ALIAS, mlflow_alias_resolver
from churn.serving.online_model import OnlineScorer, load_online_model
from churn.serving.shadow import ShadowScorer
from churn.settings import Settings

# Module-level (registered once): shadow activity is observable, not a black box.
_SHADOW_OUTCOMES = Counter(
    "churn_shadow_scores_total", "Shadow (challenger) scoring outcomes.", ["outcome"]
)


class OnlineScoreRequest(BaseModel):
    # allow_inf_nan=False: see api/serve.py CustomerPayload (REV-09); the online pipeline has no
    # domain-repair step at all, so an accepted `inf` would reach the model unclipped.
    customer_id: str
    region: str
    device_type: str
    subscription_plan: str
    account_age_days: float = Field(ge=0, allow_inf_nan=False)
    monthly_spend: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    avg_order_value: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class OnlineScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    features_source: str
    model_run_id: str | None = None


def _static_attrs(req: OnlineScoreRequest) -> dict[str, object]:
    payload = req.model_dump()
    return {col: payload[col] for col in M.STATIC_FEATURES}


def create_online_app(
    run_dir: str | None = None,
    redis_url: str | None = None,
    as_of: str | None = None,
    store: OnlineStore | None = None,
    resolver: object | None = None,
    shadow: ShadowScorer | None = None,
) -> FastAPI:
    settings = Settings.from_env()
    run_dir = run_dir or settings.online_model_run_dir
    as_of = as_of or settings.as_of
    store = store if store is not None else OnlineStore.from_url(redis_url or settings.redis_url)

    if resolver is not None:
        loader = lambda: resolver  # noqa: E731 -- injected source (tests)
    elif run_dir:
        # Pinned escape hatch: an explicit run_dir bypasses the registry on purpose.
        loader = lambda: OnlineScorer(load_online_model(run_dir))  # noqa: E731
    else:
        loader = lambda: mlflow_alias_resolver(  # noqa: E731
            settings.mlflow_model_name,
            CHAMPION_ALIAS,
            settings.mlflow_tracking_uri,
            ttl_seconds=settings.champion_ttl_seconds,
        )

    if shadow is None and resolver is None and not run_dir:
        # Shadow rides the same alias machinery as the champion (ADR 0006). No @challenger
        # aliased (or MLflow down) => per-request no-op, visible in the outcome counter.
        shadow = ShadowScorer(
            mlflow_alias_resolver(
                settings.mlflow_model_name,
                CHALLENGER_ALIAS,
                settings.mlflow_tracking_uri,
                ttl_seconds=settings.champion_ttl_seconds,
            ),
            Path(settings.shadow_log_path),
            on_outcome=lambda outcome: _SHADOW_OUTCOMES.labels(outcome).inc(),
        )

    def _score_online(
        scorer, req: OnlineScoreRequest, tasks: BackgroundTasks
    ) -> OnlineScoreResponse:
        # Strict reader (ADR 0007): a parity score is real or refused. 404 = no vector at all;
        # 409 = a vector exists but violates the contract (stale schema, wrong as_of, incomplete,
        # or a pre-envelope blob) -- NEVER silently impute from a bad payload.
        try:
            envelope = store.get(req.customer_id)
            if envelope is None:
                raise HTTPException(
                    status_code=404, detail=f"no online features for customer {req.customer_id!r}"
                )
            validate_envelope(envelope, as_of)
        except FeatureContractError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        static = _static_attrs(req)
        result = scorer.score(static, envelope.features)
        run_id = scorer.run_id
        if shadow is not None and tasks is not None:
            # Post-response, zero client latency; identical inputs so the pair is honest.
            tasks.add_task(
                shadow.score_and_log,
                customer_id=req.customer_id,
                as_of=as_of,
                static=static,
                event=envelope.features,
                champion_run_id=run_id,
                champion_probability=result.churn_probability,
            )
        return OnlineScoreResponse(
            customer_id=req.customer_id,
            churn_probability=result.churn_probability,
            risk_tier=result.risk_tier,
            features_source="redis",
            model_run_id=run_id,
        )

    return scoring_app(
        title="Churn Online Scoring",
        description="On-demand single-customer scoring from Redis online features (Sec 3).",
        path="/score/online",
        loader=loader,
        request_model=OnlineScoreRequest,
        response_model=OnlineScoreResponse,
        score_fn=_score_online,
        metrics_label="online-scoring",
    )


app = create_online_app()
