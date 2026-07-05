"""FastAPI demo for single-customer scoring; reuses the registered pipeline + tiers.

The batch-scoring endpoint. Built on the shared :func:`api.scoring_app.scoring_app` scaffold (ADR
0005): the scaffold owns load-once caching, the health/503 contract, and metrics; ``_score_batch``
owns the batch-specific work (build a one-row frame, validate it, score, report the loaded run id).
"""

from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.scoring_app import scoring_app
from churn import config, registry
from churn.data import validate_schema
from churn.scoring import ScoredCustomer


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


def _score_batch(loaded, payload: CustomerPayload) -> ScoreResponse:
    row = payload_dict(payload)
    if row.get(config.ID_COL) is None:
        row[config.ID_COL] = "api-request"
    df = pd.DataFrame([row])[[config.ID_COL, *config.RAW_FEATURE_COLUMNS]]
    validate_schema(df, require_target=False)
    record = ScoredCustomer.from_frame(loaded.score(df))[0]
    return ScoreResponse(**asdict(record), model_run_id=loaded.run_id)


def create_app(run_dir: str | None = None) -> FastAPI:
    key = run_dir or os.getenv("CHURN_MODEL_RUN_DIR")
    return scoring_app(
        title="Customer Churn Scoring Demo",
        description="Demo API; batch scoring is the primary workflow.",
        path="/score",
        loader=lambda: registry.load_run(key),
        request_model=CustomerPayload,
        response_model=ScoreResponse,
        score_fn=_score_batch,
        metrics_label="batch-scoring",
    )


app = create_app()
