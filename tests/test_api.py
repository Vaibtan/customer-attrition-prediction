"""FastAPI demo contract.

The API dependencies are optional for the core modelling environment, so these
tests skip unless the serving extra is installed.
"""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.serve import create_app  # noqa: E402
from churn import registry  # noqa: E402
from churn.data import split_features_target  # noqa: E402
from churn.pipeline import build_pipeline  # noqa: E402


def test_score_endpoint_uses_registered_model(tmp_path, sample):
    X, y, _ = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {
        "model_name": "logistic_regression",
        "threshold": 0.5,
        "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.5},
    }
    run_dir = registry.save_run(model, meta, base_model=model, base_dir=tmp_path)

    client = TestClient(create_app(str(run_dir)))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {
        "customer_id": "C-test",
        "region": "North",
        "device_type": "Mobile",
        "subscription_plan": "Free",
        "account_age_days": 500,
        "monthly_spend": 50.0,
        "num_orders_last_90d": 5,
        "avg_order_value": 100.0,
        "support_tickets_raised": 1,
        "days_since_last_login": 100,
        "pages_per_session": 8.0,
    }
    response = client.post("/score", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C-test"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert body["model_run_id"] == run_dir.name
