"""Online feature store (DIY -- Redis) -- the latest feature vector keyed by ``customer_id``.

The scoped online path (§3): FastAPI ``/score`` reads the latest online features for one customer
from Redis. The store is a thin key-value abstraction over a pluggable backend -- ``RedisBackend``
in the container, ``DictBackend`` in tests -- so the store round-trip + the online/offline parity
are testable without a live broker. Feature vectors are JSON blobs under ``feat:{customer_id}``.
"""

from __future__ import annotations

import json
from typing import Protocol

_KEY = "feat:{cid}"


class KVBackend(Protocol):
    def set(self, key: str, value: str) -> None: ...
    def get(self, key: str) -> str | None: ...


class DictBackend:
    """In-memory backend for tests + local runs (no Redis needed)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._d[key] = value

    def get(self, key: str) -> str | None:
        return self._d.get(key)


class RedisBackend:
    """redis-py backend (Phase 2 container). Imported lazily so tests never need the dependency."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        import redis  # local import: base install stays lean

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: str) -> None:
        self._r.set(key, value)

    def get(self, key: str) -> str | None:
        return self._r.get(key)


class OnlineStore:
    """Read/write the latest online feature vector for a customer."""

    def __init__(self, backend: KVBackend | None = None) -> None:
        self.backend = backend if backend is not None else DictBackend()

    @classmethod
    def from_url(cls, url: str = "redis://localhost:6379/0") -> OnlineStore:
        """Open a Redis-backed store -- the single place that wires ``OnlineStore`` to Redis."""
        return cls(RedisBackend(url))

    def put(self, customer_id: str, features: dict[str, float]) -> None:
        self.backend.set(_KEY.format(cid=customer_id), json.dumps(features, sort_keys=True))

    def put_many(self, vectors: dict[str, dict[str, float]]) -> None:
        for cid, feats in vectors.items():
            self.put(cid, feats)

    def get(self, customer_id: str) -> dict[str, float] | None:
        raw = self.backend.get(_KEY.format(cid=customer_id))
        return None if raw is None else json.loads(raw)
