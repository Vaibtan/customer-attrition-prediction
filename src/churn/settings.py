"""Runtime settings: every deployment env knob in one typed, frozen place (12-factor).

The split is deliberate (CONTEXT.md: *runtime settings vs policy*): this module owns the knobs
that vary per deployment (URLs, topics, paths, timeouts) and nothing else. Frozen ML *policy*
(thresholds, MDEs, tolerances) lives in :mod:`churn.config` and is deliberately NOT
env-overridable -- tuned once, frozen with the policy it implements.

Zero dependencies by design: a frozen dataclass + ``from_env`` covers what this project needs
(string/float knobs, explicit env names, injectable mapping for tests) without adding
pydantic-settings to the core package.

Usage: ``settings = Settings.from_env()`` at an entrypoint; pass values down as parameters.
``.env.example`` at the repo root enumerates every knob.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import ClassVar


@dataclass(frozen=True)
class Settings:
    """Deployment-varying runtime configuration, resolved once from the environment."""

    redis_url: str = "redis://localhost:6379/0"
    redpanda_broker: str = "localhost:9092"
    events_topic: str = "customer-events"
    consumer_group: str = "churn-streaming-v1"
    events_parquet: str = "data/stream/events.parquet"
    as_of: str = "2025-01-01T00:00:00"
    state_dir: str = "/tmp/quix-state"
    consumer_timeout: float = 0.0
    model_run_dir: str | None = None
    online_model_run_dir: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_model_name: str = "churn-online"
    champion_ttl_seconds: float = 60.0
    shadow_log_path: str = "monitoring/shadow_scores.jsonl"

    # Explicit env names: several predate this module (compose/CI reference them), so they are a
    # contract -- do not derive them from field names.
    _ENV: ClassVar[dict[str, str]] = {
        "redis_url": "REDIS_URL",
        "redpanda_broker": "REDPANDA_BROKER",
        "events_topic": "EVENTS_TOPIC",
        "consumer_group": "CONSUMER_GROUP",
        "events_parquet": "EVENTS_PARQUET",
        "as_of": "CHURN_AS_OF",
        "state_dir": "CHURN_STATE_DIR",
        "consumer_timeout": "CONSUMER_TIMEOUT",
        "model_run_dir": "CHURN_MODEL_RUN_DIR",
        "online_model_run_dir": "CHURN_ONLINE_MODEL_RUN_DIR",
        "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
        "mlflow_model_name": "MLFLOW_MODEL_NAME",
        "champion_ttl_seconds": "CHAMPION_TTL_SECONDS",
        "shadow_log_path": "SHADOW_LOG_PATH",
    }
    _CAST: ClassVar[dict[str, type]] = {"consumer_timeout": float, "champion_ttl_seconds": float}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build from ``env`` (default ``os.environ``); unset vars keep the field default."""
        src = os.environ if env is None else env
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = src.get(cls._ENV[f.name])
            if raw is not None:
                kwargs[f.name] = cls._CAST.get(f.name, str)(raw)
        return cls(**kwargs)
