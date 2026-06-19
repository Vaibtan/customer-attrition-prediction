"""FastAPI demo for single-customer scoring; reuses the registered pipeline + tiers."""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn import config, registry
from churn.data import validate_schema


class CustomerPayload(BaseModel):
    customer_id: str | None = None
    region: str
    device_type: str
    subscription_plan: str
    account_age_days: float = Field(ge=0)
    monthly_spend: float | None = Field(default=None, ge=0)
    num_orders_last_90d: float = Field(ge=0)
    avg_order_value: float | None = Field(default=None, ge=0)
    support_tickets_raised: float = Field(ge=0)
    days_since_last_login: float = Field(ge=0)
    pages_per_session: float | None = Field(default=None, ge=0)


class ScoreResponse(BaseModel):
    customer_id: str | int | None
    churn_probability: float
    risk_tier: str
    top_reason_codes: str | None = None
    model_run_id: str


def payload_dict(payload: CustomerPayload) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def state_key(run_dir: str | None) -> str:
    return run_dir or "__latest__"


@lru_cache(maxsize=8)
def load_state(run_dir_key: str):
    run_dir = None if run_dir_key == "__latest__" else run_dir_key
    return registry.load_run(run_dir)


def create_app(run_dir: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Customer Churn Scoring Demo",
        version="0.1.0",
        description="Demo API; batch scoring is the primary workflow.",
    )
    key = state_key(run_dir or os.getenv("CHURN_MODEL_RUN_DIR"))

    @app.get("/health")
    def health():
        try:
            loaded = load_state(key)
        except FileNotFoundError:
            return {"status": "degraded", "model_loaded": False}
        return {"status": "ok", "model_loaded": True, "run_id": loaded.run_id}

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: CustomerPayload):
        try:
            loaded = load_state(key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        row = payload_dict(payload)
        if row.get(config.ID_COL) is None:
            row[config.ID_COL] = "api-request"
        df = pd.DataFrame([row])[[config.ID_COL, *config.RAW_FEATURE_COLUMNS]]
        validate_schema(df, require_target=False)

        scored = loaded.score(df).iloc[0]
        return {
            "customer_id": scored[config.ID_COL],
            "churn_probability": float(scored["churn_probability"]),
            "risk_tier": scored["risk_tier"],
            "top_reason_codes": scored.get("top_reason_codes"),
            "model_run_id": loaded.run_id,
        }

    return app


app = create_app()
