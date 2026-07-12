"""Runtime Settings: typed, frozen, env-driven with explicit names (12-factor)."""

from __future__ import annotations

import dataclasses

import pytest

from churn.settings import Settings


def test_defaults_apply_when_env_is_empty():
    s = Settings.from_env(env={})
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.redpanda_broker == "localhost:9092"
    assert s.consumer_timeout == 0.0
    assert s.model_run_dir is None


def test_env_overrides_with_the_contract_names_and_casts():
    s = Settings.from_env(
        env={
            "REDIS_URL": "redis://prod:6379/1",
            "REDPANDA_BROKER": "broker:9092",
            "CHURN_AS_OF": "2025-06-01T00:00:00",
            "CONSUMER_TIMEOUT": "12.5",
            "CHURN_MODEL_RUN_DIR": "models/some-run",
        }
    )
    assert s.redis_url == "redis://prod:6379/1"
    assert s.redpanda_broker == "broker:9092"
    assert s.as_of == "2025-06-01T00:00:00"
    assert s.consumer_timeout == 12.5  # cast, not the string
    assert s.model_run_dir == "models/some-run"
    assert s.events_topic == "customer-events"  # unset -> default


def test_settings_are_frozen():
    s = Settings.from_env(env={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.redis_url = "redis://other"  # type: ignore[misc]


def test_every_field_has_an_env_name():
    assert set(Settings._ENV) == {f.name for f in dataclasses.fields(Settings)}


def test_env_example_documents_every_knob():
    """.env.example is the operator contract -- it must enumerate every env name."""
    from churn import config  # noqa: PLC0415

    text = (config.ROOT / ".env.example").read_text()
    for env_name in Settings._ENV.values():
        assert env_name in text, f"{env_name} missing from .env.example"
