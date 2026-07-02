"""Shared fixtures and builders for the test suite."""

from __future__ import annotations

import os
import uuid

import pandas as pd
import pytest

from churn import config
from churn.data import load_data

DEFAULTS = {
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


def make_raw(**overrides) -> pd.DataFrame:
    row = {**DEFAULTS, **overrides}
    return pd.DataFrame([row])[config.RAW_FEATURE_COLUMNS]


@pytest.fixture(scope="session")
def raw_full() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="session")
def sample(raw_full) -> pd.DataFrame:
    return raw_full.sample(n=300, random_state=config.SEED).reset_index(drop=True)


# --- Real-infra fixtures (integration tests) --------------------------------------------------
# These wire tests to LIVE Docker services (Redpanda, Redis). They read connection info from env
# and SKIP when it is absent, so the default host run stays fast and infra-free; the real path is
# exercised inside the `test-runner` container (`docker compose --profile test run --rm
# test-runner`), where the env vars + the Linux-only clients are present. Clients are imported
# lazily so host collection never needs confluent-kafka / redis installed.


def _env_or_skip(var: str) -> str:
    value = os.getenv(var)
    if not value:
        pytest.skip(
            f"{var} unset -- integration test needs live infra "
            f"(run: docker compose --profile test run --rm test-runner)"
        )
    return value


@pytest.fixture
def redpanda_broker() -> str:
    return _env_or_skip("REDPANDA_BROKER")


@pytest.fixture
def redis_url() -> str:
    return _env_or_skip("REDIS_URL")


@pytest.fixture
def redis_client(redis_url: str):
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


@pytest.fixture
def kafka_topic(redpanda_broker: str):
    """Create a uniquely-named topic on the live broker; drop it on teardown."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": redpanda_broker})
    name = f"it-events-{uuid.uuid4().hex[:12]}"
    created = admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=1)])
    created[name].result(timeout=30)
    try:
        yield name
    finally:
        try:
            deleted = admin.delete_topics([name])
            deleted[name].result(timeout=30)
        except Exception:  # noqa: BLE001 -- best-effort cleanup; a leaked test topic is harmless
            pass


@pytest.fixture
def unique_group() -> str:
    """A fresh consumer group per test so `auto_offset_reset=earliest` always replays from 0."""
    return f"it-group-{uuid.uuid4().hex[:12]}"
