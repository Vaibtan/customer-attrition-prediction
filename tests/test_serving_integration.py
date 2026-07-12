"""Online serving parity against LIVE Redpanda + Redis + FastAPI (score-level train/serve check).

Stream a cohort's events to real Redpanda -> real Quix consumer -> real Redis, then call the online
``/score/online`` endpoint (reading Redis) and assert the returned probability equals the OFFLINE
batch score for the same customer. Since online event features == offline PIT features (parity), the
served online score must equal the batch score -- the train/serve-consistency guarantee at the score
level, through the real deployment path.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("confluent_kafka")
pytest.importorskip("quixstreams")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.online import create_online_app  # noqa: E402
from churn.featurestore import offline as OFF  # noqa: E402
from churn.featurestore.cohort import make_cohort  # noqa: E402
from churn.featurestore.online import OnlineStore  # noqa: E402
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
    train_cohort = make_cohort(train_ids, T0)
    train_pit = OFF.compute_pit_features(ds.events, train_cohort)
    model = OM.train_online_model(ds, train_pit, seed=42)
    run_dir = OM.save_online_model(model, base_dir=tmp_path)

    # Stream this cohort's events (shuffled) into Redis via the real broker + consumer.
    PROD.produce_events(redpanda_broker, kafka_topic, events.sample(frac=1.0, random_state=2))
    store = OnlineStore.from_url(redis_url)
    CONSUMER.run_consumer(
        broker=redpanda_broker,
        topic=kafka_topic,
        consumer_group=unique_group,
        t0=T0,
        online_store=store,
        timeout=10.0,
        state_dir=str(tmp_path / "quix-state"),
    )

    cohort = make_cohort(ids, T0)
    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")
    stat = ds.customers.set_index("customer_id")
    scorer = OM.OnlineScorer(model)

    client = TestClient(create_online_app(run_dir=str(run_dir), redis_url=redis_url))
    health = client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["run_id"] == run_dir.name  # ISS-04: the real registry run id, not a config path

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
        assert body["model_run_id"] == run_dir.name  # ISS-04: real run id threaded to the response
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

    # ADR 0007 strict reader through the live store: a stale-schema envelope is refused (409),
    # never silently imputed into a "parity" score.
    stale = {
        "schema": "deadbeef0000",
        "as_of": T0.isoformat(),
        "features": {col: 1.0 for col in OFF.FEATURE_COLUMNS},
    }
    store.backend.set("feat:STALE-SCHEMA", json.dumps(stale))
    conflict = client.post(
        "/score/online",
        json={
            "customer_id": "STALE-SCHEMA",
            "region": "North",
            "device_type": "Mobile",
            "subscription_plan": "Free",
            "account_age_days": 100.0,
        },
    )
    assert conflict.status_code == 409
    assert "schema mismatch" in conflict.json()["detail"]


def test_champion_cutover_and_shadow_against_live_mlflow(mlflow_uri, tmp_path):
    """ADR 0006 end-to-end on a real tracking server: the app follows @champion (promotion IS
    the cutover), health reports alias provenance, and @challenger shadow-scores post-response."""
    import json as _json  # noqa: PLC0415

    from churn.featurestore.online import OnlineStore  # noqa: PLC0415
    from churn.lifecycle import mlflow_registry as MR  # noqa: PLC0415
    from churn.serving.champion import mlflow_alias_resolver  # noqa: PLC0415
    from churn.serving.shadow import ShadowScorer  # noqa: PLC0415
    from churn.streaming import aggregate as AGG  # noqa: PLC0415

    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = _cohort_with_events_and_static(ds, 3)
    events = ds.events[ds.events["customer_id"].isin(ids)]

    train_ids = ds.customers["customer_id"].head(500).tolist()
    train_pit = OFF.compute_pit_features(ds.events, make_cohort(train_ids, T0))
    model = OM.train_online_model(ds, train_pit, seed=42)
    name = f"churn-online-it-{tmp_path.name.lower()}"
    tags = {"tier_cutpoints": _json.dumps(model.cutpoints)}

    v1 = MR.log_and_register(model.pipeline, name, {"roc_auc": 0.7}, mlflow_uri, tags=tags)
    MR.set_alias(name, MR.CHAMPION, v1, mlflow_uri)
    MR.set_alias(name, MR.CHALLENGER, v1, mlflow_uri)  # shadow scores with v1 too

    store = OnlineStore()  # local backend: this test isolates the MLflow leg
    store.put_many(AGG.aggregate_stream(events, {c: T0 for c in ids}), as_of=T0)

    shadow_log = tmp_path / "shadow.jsonl"
    app = create_online_app(
        as_of=str(T0),
        store=store,
        resolver=mlflow_alias_resolver(name, MR.CHAMPION, mlflow_uri, ttl_seconds=0.0),
        shadow=ShadowScorer(
            mlflow_alias_resolver(name, MR.CHALLENGER, mlflow_uri, ttl_seconds=0.0), shadow_log
        ),
    )
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["run_id"] == f"{name}@v{v1}"
    assert health["alias_version"] == v1

    stat = ds.customers.set_index("customer_id")
    c = ids[0]
    payload = {
        "customer_id": c,
        "region": stat.loc[c, "region"],
        "device_type": stat.loc[c, "device_type"],
        "subscription_plan": stat.loc[c, "subscription_plan"],
        "account_age_days": _num(stat.loc[c, "account_age_days"]),
        "monthly_spend": _num(stat.loc[c, "monthly_spend"]),
        "avg_order_value": _num(stat.loc[c, "avg_order_value"]),
    }
    resp = client.post("/score/online", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["model_run_id"] == f"{name}@v{v1}"

    # TestClient runs background tasks before returning: the shadow record is already on disk.
    record = _json.loads(shadow_log.read_text().splitlines()[-1])
    assert record["customer_id"] == c
    assert record["challenger_run_id"] == f"{name}@v{v1}"
    assert record["champion_probability"] == pytest.approx(resp.json()["churn_probability"])

    # Promotion IS the cutover: register v2, move @champion, the next request serves it.
    v2 = MR.log_and_register(model.pipeline, name, {"roc_auc": 0.8}, mlflow_uri, tags=tags)
    MR.promote(name, v2, mlflow_uri)
    assert client.post("/score/online", json=payload).json()["model_run_id"] == f"{name}@v{v2}"
    assert client.get("/health").json()["alias_version"] == v2
