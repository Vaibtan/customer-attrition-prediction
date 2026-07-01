"""Confirmatory-seed & floor-RNG derivation recipe (PHASE0_LOCK_DECISIONS.md D3/D4).

Phase 0 commits the *recipe* + a verifiable derivation and its tests. There is **no live beacon
fetch and no confirmatory run here** -- those are Stage 2 / end of Phase 1. The confirmatory seed
and every pass/fail-determining floor RNG derive from a single pinned drand public-randomness
beacon round ``R`` via domain-separated tags, so they are neither author-chosen nor knowable before
the analysis is frozen.

Pinned chain: drand League-of-Entropy **quicknet** (unchained, 3s, BLS-on-G1, RFC-9380). Fetched
from https://api.drand.sh/<hash>/info; recorded here and in ``simulator.lock.json`` so "the beacon"
itself cannot be swapped. Golden vectors (``tests/test_simulator_beacon.py``) pin the round rule +
KDF against hand-computed values with placeholder randomness.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# --- Pinned drand quicknet chain parameters (recorded in the lock too) -----------------------
CHAIN = {
    "scheme": "drand-quicknet",
    "beacon_id": "quicknet",
    "chain_hash": "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
    "public_key": (
        "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
        "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
        "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
    ),
    "genesis_time": 1692803367,
    "period_seconds": 3,
    "scheme_id": "bls-unchained-g1-rfc9380",
    "group_hash": "f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e",
}

# --- Round rule + CI-enforcement constants ---------------------------------------------------
# R = the first chain round whose beacon time >= (T_trusted + DELTA_SECONDS). T_trusted is the
# protected-branch merge / CI first-seen time of the Stage-2 lock -- NEVER the author's git date.
# R is >= 1h after T_trusted -> unemitted (hence unknowable) at lock time.
DELTA_SECONDS = 3600
# CI rejects a Stage-2 commit whose git date is outside this skew of T_trusted.
COMMIT_SKEW_SECONDS = 900

# --- Domain-separated KDF tags (D3/D4) -------------------------------------------------------
# A dedicated tag for EVERY pass/fail-determining RNG stream, so none is ambiguous or shop-able:
# the confirmatory data draw + its paired bootstrap, the oracle/MDE/floor draws, and both
# confirmatory negative controls.
DOMAIN_TAGS = {
    "confirmatory": "churn-confirmatory",
    "confirmatory_bootstrap": "churn-confirmatory-bootstrap",
    "oracle": "churn-oracle",
    "mde": "churn-mde",
    "floor_bootstrap": "churn-floor-bootstrap",
    "control_label_shuffle": "churn-control-label-shuffle",
    "control_null_stream": "churn-control-null-stream",
}

KDF_DESCRIPTION = (
    "seed_int = int.from_bytes(sha256(domain_tag + ':' + randomness_hex).digest(), 'big')"
)
ROUND_RULE_DESCRIPTION = (
    "R = first round with time >= (T_trusted + DELTA_SECONDS); time(r) = genesis + (r-1)*period"
)


def round_at_or_after(t_target_unix: int) -> int:
    """First chain round whose beacon time is >= ``t_target_unix`` (integer, exact)."""
    genesis = CHAIN["genesis_time"]
    period = CHAIN["period_seconds"]
    delta = int(t_target_unix) - genesis
    ceil_div = -((-delta) // period)  # integer ceil, valid for any sign
    return max(1, 1 + ceil_div)


def confirmatory_round(t_trusted_unix: int) -> int:
    """The pinned round R keyed to the *trusted* clock: first round >= T_trusted + DELTA_SECONDS."""
    return round_at_or_after(int(t_trusted_unix) + DELTA_SECONDS)


def _kdf_digest(randomness_hex: str, domain_tag: str) -> bytes:
    # drand quicknet randomness is 32 bytes = 64 lowercase hex chars; reject anything else so a
    # malformed / attacker-supplied randomness string cannot silently produce a "valid" seed.
    if not re.fullmatch(r"[0-9a-f]{64}", randomness_hex):
        raise ValueError("randomness_hex must be exactly 64 lowercase hex chars (32-byte beacon)")
    if domain_tag not in DOMAIN_TAGS.values():
        raise ValueError(f"unknown domain tag: {domain_tag!r}")
    return hashlib.sha256(f"{domain_tag}:{randomness_hex}".encode()).digest()


def derive_seed_int(randomness_hex: str, domain_tag: str) -> int:
    """Domain-separated seed integer from a round's randomness (256-bit, big-endian)."""
    return int.from_bytes(_kdf_digest(randomness_hex, domain_tag), "big")


def derive_seed_hex(randomness_hex: str, domain_tag: str) -> str:
    return _kdf_digest(randomness_hex, domain_tag).hex()


def seed_sequence(randomness_hex: str, domain_tag: str) -> np.random.SeedSequence:
    """NumPy SeedSequence for the given round randomness + domain tag (exact byte order pinned)."""
    return np.random.SeedSequence(derive_seed_int(randomness_hex, domain_tag))


def recipe() -> dict:
    """The beacon section embedded verbatim into ``simulator.lock.json``."""
    return {
        **CHAIN,
        "delta_seconds": DELTA_SECONDS,
        "commit_skew_seconds": COMMIT_SKEW_SECONDS,
        "round_rule": ROUND_RULE_DESCRIPTION,
        "kdf": KDF_DESCRIPTION,
        "domain_tags": DOMAIN_TAGS,
        "t_trusted_rule": (
            "protected-branch merge / CI first-seen time of the Stage-2 lock; NEVER the author's "
            "git commit/committer date. CI rejects backdated commits (skew window), requires R "
            "unemitted at first check, recomputes R, and rejects any other R or seed."
        ),
        "note": (
            "Phase 0 commits this recipe only -- no live fetch, no confirmatory run until Stage 2 "
            "(D3/D4). A sealed-hash commitment is reserved for genuine multi-party custody and is "
            "NOT used here."
        ),
    }
