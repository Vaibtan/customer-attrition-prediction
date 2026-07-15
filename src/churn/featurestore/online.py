"""Online feature store (DIY -- Redis) -- versioned feature envelopes keyed by ``customer_id``.

The scoped online path (§3): FastAPI ``/score/online`` reads the latest online features for one
customer from Redis. The store is a thin key-value abstraction over a pluggable backend --
``RedisBackend`` in the container, ``DictBackend`` in tests -- so the store round-trip + the
online/offline parity are testable without a live broker.

**The payload is a contract, not a bare dict (ADR 0007):** every write is an envelope
``{schema, as_of, features}`` under ``feat:{customer_id}``.

- ``schema`` is DERIVED from the canonical feature-column spec (``offline.FEATURE_COLUMNS``), so
  it changes by construction when the spec changes -- no manual bump to forget (the ISS-09
  lesson, applied to the wire format).
- ``as_of`` is the scoring instant the vector was computed for. In the frozen world freshness
  means *as_of equality* with the scorer's configured instant, never a wall-clock TTL.
- Writes are ``allow_nan=False``: a NaN feature fails loudly at the producer instead of
  poisoning the store (bounds the REV-04 class of latent divergence).
- The reader is STRICT (:func:`validate_envelope`): schema/as_of mismatch or an incomplete
  vector raises :class:`FeatureContractError` -- parity scores are real or refused, never
  imputed from a half-written blob.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from churn.featurestore import offline as OFF

_KEY = "feat:{cid}"

# Derived, not hand-bumped: sha256 of the canonical feature-column list, first 12 hex chars.
FEATURE_SCHEMA: str = hashlib.sha256("|".join(OFF.FEATURE_COLUMNS).encode("ascii")).hexdigest()[:12]


class FeatureContractError(ValueError):
    """The stored payload violates the online feature contract (ADR 0007)."""


def _normalize_as_of(as_of: object) -> str:
    """Canonical ``as_of`` wire form so producer/reader format quirks can't fake a mismatch."""
    return pd.Timestamp(as_of).isoformat()


@dataclass(frozen=True)
class FeatureEnvelope:
    """One customer's versioned online feature vector."""

    schema: str
    as_of: str
    features: dict[str, float]


def validate_envelope(envelope: FeatureEnvelope, expected_as_of: object) -> None:
    """Strict reader checks: raise :class:`FeatureContractError` on any contract violation."""
    if envelope.schema != FEATURE_SCHEMA:
        raise FeatureContractError(
            f"feature schema mismatch: stored {envelope.schema!r}, expected {FEATURE_SCHEMA!r}"
            " (stale vector from an older feature spec -- rebuild the online store)"
        )
    expected = _normalize_as_of(expected_as_of)
    if envelope.as_of != expected:
        raise FeatureContractError(
            f"as_of mismatch: stored {envelope.as_of!r}, expected {expected!r}"
            " (vector computed for a different scoring instant)"
        )
    missing = [c for c in OFF.FEATURE_COLUMNS if c not in envelope.features]
    if missing:
        raise FeatureContractError(f"incomplete feature vector: missing {missing}")


class KVBackend(Protocol):
    def set(self, key: str, value: str) -> None: ...
    def set_many(self, items: dict[str, str]) -> None: ...
    def get(self, key: str) -> str | None: ...


class DictBackend:
    """In-memory backend for tests + local runs (no Redis needed)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._d[key] = value

    def set_many(self, items: dict[str, str]) -> None:
        self._d.update(items)

    def get(self, key: str) -> str | None:
        return self._d.get(key)


class RedisBackend:
    """redis-py backend (Phase 2 container). Imported lazily so tests never need the dependency."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        import redis  # local import: base install stays lean

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: str) -> None:
        self._r.set(key, value)

    def set_many(self, items: dict[str, str]) -> None:
        self._r.mset(items)  # one round trip for the whole batch (ADR 0008)

    def get(self, key: str) -> str | None:
        return self._r.get(key)


def _envelope_json(features: dict[str, float], as_of: str) -> str:
    """Serialize one feature envelope. allow_nan=False fails at the producer instead of persisting
    a NaN feature -- bounds the REV-04 class of latent divergence for both single and batch writes.
    """
    payload = {"schema": FEATURE_SCHEMA, "as_of": as_of, "features": features}
    return json.dumps(payload, sort_keys=True, allow_nan=False)


class OnlineStore:
    """Read/write the latest online feature envelope for a customer."""

    def __init__(self, backend: KVBackend | None = None) -> None:
        self.backend = backend if backend is not None else DictBackend()

    @classmethod
    def from_url(cls, url: str = "redis://localhost:6379/0") -> OnlineStore:
        """Open a Redis-backed store -- the single place that wires ``OnlineStore`` to Redis."""
        return cls(RedisBackend(url))

    def put(self, customer_id: str, features: dict[str, float], as_of: object) -> None:
        self.backend.set(
            _KEY.format(cid=customer_id), _envelope_json(features, _normalize_as_of(as_of))
        )

    def put_many(self, vectors: dict[str, dict[str, float]], as_of: object) -> None:
        # Build ALL envelopes first, then one batched backend call (ADR 0008): the "rebuild the
        # online store" path promised by the contract errors must not inherit an N-round-trip
        # footgun. Encoding up front also makes the batch all-or-nothing -- a single NaN vector
        # raises before anything reaches the store, never a partial write.
        as_of = _normalize_as_of(as_of)
        items = {
            _KEY.format(cid=cid): _envelope_json(feats, as_of) for cid, feats in vectors.items()
        }
        if not items:  # redis MSET rejects an empty mapping; an empty batch is a no-op
            return
        self.backend.set_many(items)

    def get(self, customer_id: str) -> FeatureEnvelope | None:
        """Read one envelope; ``None`` on a missing key, contract error on a legacy/foreign blob."""
        raw = self.backend.get(_KEY.format(cid=customer_id))
        if raw is None:
            return None
        doc = json.loads(raw)
        if not isinstance(doc, dict) or not {"schema", "as_of", "features"} <= set(doc):
            raise FeatureContractError(
                "unversioned online payload (pre-envelope blob) -- rebuild the online store"
            )
        return FeatureEnvelope(schema=doc["schema"], as_of=doc["as_of"], features=doc["features"])
