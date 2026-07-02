"""Online serving parity against LIVE Redpanda + Redis + FastAPI (score-level train/serve check).

Stream a cohort's events to real Redpanda -> real Quix consumer -> real Redis, then call the online
``/score/online`` endpoint (reading Redis) and assert the returned probability equals the OFFLINE
batch score for the same customer. Since online event features == offline PIT features (parity), the
served online score must equal the batch score -- the train/serve-consistency guarantee at the score
level, through the real deployment path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("confluent_kafka")
pytest.importorskip("quixstreams")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.online import create_online_app  # noqa: E402
from churn.featurestore import offline as OFF  # noqa: E402
from churn.featurestore.online import OnlineStore, RedisBackend  # noqa: E402
from churn.serving import online_model as OM  # noqa: E402
from churn.simulator import generate as G  # noqa: E402
from churn.simulator import params as P  # noqa: E402
from churn.streaming import producer as PROD  # noqa: E402
from services.consumer import app as CONSUMER  # noqa: E402

pytestmark = pytest.mark.integration

T0 = pd.Timestamp("2025-01-01T00:00:00")


def _num(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def _cohort_with_events_and_static(ds, n: int) -> list[str]:
    counts = ds.events["customer_id"].value_counts()
    stat = ds.customers.set_index("customer_id")
    have = [
        c
        for c in ds.customers["customer_id"].tolist()
        if counts.get(c, 0) > 0 and not pd.isna(stat.loc[c, "account_age_days"])
    ]
    return have[:n]


def test_online_score_equals_offline_batch_score(
    redpanda_broker, redis_url, kafka_topic, unique_group, redis_client, tmp_path
):
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _cohort_with_events_and_static(ds, 20)
    events = ds.events[ds.events["customer_id"].isin(ids)].reset_index(drop=True)

    # Train + register the deployable online model on a broad cohort (stable fit).
    train_ids = ds.customers["customer_id"].head(500).tolist()
    train_cohort = pd.DataFrame({"customer_id": train_ids, "t0": [T0] * len(train_ids)})
    train_pit = OFF.compute_pit_features(ds.events, train_cohort)
    model = OM.train_online_model(ds, train_pit, seed=42)
    run_dir = OM.save_online_model(model, base_dir=tmp_path)

    # Stream this cohort's events (shuffled) into Redis via the real broker + consumer.
    PROD.produce_events(redpanda_broker, kafka_topic, events.sample(frac=1.0, random_state=2))
    store = OnlineStore(RedisBackend(redis_url))
    CONSUMER.run_consumer(
        broker=redpanda_broker,
        topic=kafka_topic,
        consumer_group=unique_group,
        t0=T0,
        online_store=store,
        timeout=10.0,
        state_dir=str(tmp_path / "quix-state"),
    )

    cohort = pd.DataFrame({"customer_id": ids, "t0": [T0] * len(ids)})
    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")
    stat = ds.customers.set_index("customer_id")
    scorer = OM.OnlineScorer(model)

    client = TestClient(create_online_app(run_dir=str(run_dir), redis_url=redis_url))
    assert client.get("/health").json()["model_loaded"] is True

    for c in ids:
        static = {
            "region": stat.loc[c, "region"],
            "device_type": stat.loc[c, "device_type"],
            "subscription_plan": stat.loc[c, "subscription_plan"],
            "account_age_days": _num(stat.loc[c, "account_age_days"]),
            "monthly_spend": _num(stat.loc[c, "monthly_spend"]),
            "avg_order_value": _num(stat.loc[c, "avg_order_value"]),
        }
        offline_event = {col: float(offline.loc[c, col]) for col in OFF.FEATURE_COLUMNS}
        expected = scorer.score(static, offline_event).churn_probability

        resp = client.post("/score/online", json={"customer_id": c, **static})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["features_source"] == "redis"
        assert body["churn_probability"] == pytest.approx(expected, abs=1e-9)
        assert body["risk_tier"] in {"low", "medium", "high"}

    missing = client.post(
        "/score/online",
        json={
            "customer_id": "GHOST-NO-EVENTS",
            "region": "North",
            "device_type": "Mobile",
            "subscription_plan": "Free",
            "account_age_days": 100.0,
        },
    )
    assert missing.status_code == 404
