"""On-demand online scoring API: ``POST /score/online`` reads Redis features + request statics.

The scoped online path (ENHANCEMENT_PLAN.md Sec 3): a CSM/UI pulls up one customer; the endpoint
reads the latest event-feature vector from Redis (written by the streaming consumer, keyed by
``customer_id``) and combines it with the static attributes in the request, scoring with the
deployed static+event model. Online features == offline PIT features (parity), so this equals the
offline batch score for the same ``(customer, t0)`` -- train/serve consistency at the score level.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.featurestore.online import OnlineStore, RedisBackend
from churn.instrument import model as M
from churn.serving.online_model import OnlineScorer, load_online_model


class OnlineScoreRequest(BaseModel):
    customer_id: str
    region: str
    device_type: str
    subscription_plan: str
    account_age_days: float = Field(ge=0)
    monthly_spend: float | None = Field(default=None, ge=0)
    avg_order_value: float | None = Field(default=None, ge=0)


class OnlineScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    features_source: str
    model_run_id: str | None = None


def _static_attrs(req: OnlineScoreRequest) -> dict[str, object]:
    payload = req.model_dump()
    return {col: payload[col] for col in M.STATIC_FEATURES}


def create_online_app(run_dir: str | None = None, redis_url: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Churn Online Scoring",
        version="0.1.0",
        description="On-demand single-customer scoring from Redis online features (Sec 3).",
    )
    run_dir = run_dir or os.getenv("CHURN_ONLINE_MODEL_RUN_DIR")
    redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    store = OnlineStore(RedisBackend(redis_url))
    state: dict[str, object] = {}

    def _scorer() -> OnlineScorer:
        if "scorer" not in state:
            state["scorer"] = OnlineScorer(load_online_model(run_dir))
        return state["scorer"]  # type: ignore[return-value]

    @app.get("/health")
    def health():
        try:
            _scorer()
        except FileNotFoundError:
            return {"status": "degraded", "model_loaded": False}
        return {"status": "ok", "model_loaded": True}

    @app.post("/score/online", response_model=OnlineScoreResponse)
    def score_online(req: OnlineScoreRequest):
        try:
            scorer = _scorer()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        event = store.get(req.customer_id)
        if event is None:
            raise HTTPException(
                status_code=404, detail=f"no online features for customer {req.customer_id!r}"
            )
        result = scorer.score(_static_attrs(req), event)
        return OnlineScoreResponse(
            customer_id=req.customer_id,
            churn_probability=result.churn_probability,
            risk_tier=result.risk_tier,
            features_source="redis",
            model_run_id=str(run_dir) if run_dir else None,
        )

    return app


app = create_online_app()
