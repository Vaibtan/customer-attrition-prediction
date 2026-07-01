"""Golden vectors for the confirmatory-seed / floor-RNG beacon recipe (D3/D4).

Phase 0 commits the recipe + a *verifiable derivation* and its tests -- no live fetch. These pin
the round rule and the domain-separated KDF against hand-computed values with placeholder
randomness ("ab"*32, NOT a real beacon draw), so a silent change to the derivation is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.simulator import beacon

GENESIS = 1692803367  # drand quicknet genesis
RAND = "ab" * 32  # 64-hex placeholder randomness (not a real round value)

# Precomputed sha256(tag + ":" + RAND).hex() -- the frozen KDF outputs.
EXPECTED_SEED_HEX = {
    "churn-confirmatory": "61b784166e374e64c47c8e0aaa44f60e558beb2446cd137225562449e169b4ed",
    "churn-oracle": "88eb8dee9ebdfdc5d8a8068a25b365cd1e42b33c5b15586cb0064d7db370ee20",
    "churn-mde": "d06e765f539ceda4396cd9f88efd831cbd283041206ff8e5c38bba249c814114",
    "churn-floor-bootstrap": "be5ef926e5a7e0449c7d8e632e13b9712f3e29e6cd15806cd0bcecc843000b74",
}


# --- round rule (D3) -------------------------------------------------------------------------


def test_round_at_or_after_is_the_declared_rule():
    # time(r) = genesis + (r-1)*period, period = 3s.
    assert beacon.round_at_or_after(GENESIS) == 1  # round 1 at genesis
    assert beacon.round_at_or_after(GENESIS + 1) == 2  # round 1 too early
    assert beacon.round_at_or_after(GENESIS + 3) == 2  # round 2 exactly at genesis+3
    assert beacon.round_at_or_after(GENESIS + 4) == 3
    assert beacon.round_at_or_after(GENESIS + 6) == 3
    assert beacon.round_at_or_after(GENESIS - 100) == 1  # never below round 1


def test_confirmatory_round_uses_delta_after_trusted_clock():
    t_trusted = 1_700_000_000
    # R = round_at_or_after(t_trusted + DELTA_SECONDS); DELTA = 3600.
    assert beacon.confirmatory_round(t_trusted) == beacon.round_at_or_after(t_trusted + 3600)
    assert beacon.confirmatory_round(t_trusted) == 2_400_079


# --- KDF + domain separation (D3/D4) ---------------------------------------------------------


def test_kdf_matches_frozen_golden_hex():
    for tag, expected in EXPECTED_SEED_HEX.items():
        assert beacon.derive_seed_hex(RAND, tag) == expected
        assert beacon.derive_seed_int(RAND, tag) == int(expected, 16)


def test_domain_tags_are_separated():
    seeds = {beacon.derive_seed_int(RAND, t) for t in beacon.DOMAIN_TAGS.values()}
    assert len(seeds) == len(beacon.DOMAIN_TAGS)  # all distinct


def test_seed_sequence_is_reproducible_and_tag_dependent():
    ss_a = beacon.seed_sequence(RAND, "churn-confirmatory")
    ss_b = beacon.seed_sequence(RAND, "churn-confirmatory")
    draw_a = np.random.default_rng(ss_a).random(5)
    draw_b = np.random.default_rng(ss_b).random(5)
    assert np.array_equal(draw_a, draw_b)

    ss_c = beacon.seed_sequence(RAND, "churn-oracle")
    draw_c = np.random.default_rng(ss_c).random(5)
    assert not np.array_equal(draw_a, draw_c)


# --- recipe pins the chain (D3: "the beacon itself cannot be swapped") ------------------------


def test_recipe_pins_the_quicknet_chain():
    r = beacon.recipe()
    assert r["chain_hash"] == "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
    assert r["genesis_time"] == GENESIS
    assert r["period_seconds"] == 3
    assert r["scheme_id"] == "bls-unchained-g1-rfc9380"
    assert r["delta_seconds"] == 3600
    # Every pass/fail-determining stream has a dedicated, non-shoppable tag.
    assert set(r["domain_tags"]) == {
        "confirmatory",
        "confirmatory_bootstrap",
        "oracle",
        "mde",
        "floor_bootstrap",
        "control_label_shuffle",
        "control_null_stream",
    }


def test_kdf_rejects_malformed_randomness_and_unknown_tags():
    with pytest.raises(ValueError, match="64 lowercase hex"):
        beacon.derive_seed_int("not-hex", "churn-confirmatory")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        beacon.derive_seed_int("ab" * 31, "churn-confirmatory")  # too short
    with pytest.raises(ValueError, match="unknown domain tag"):
        beacon.derive_seed_int(RAND, "churn-not-a-real-tag")
