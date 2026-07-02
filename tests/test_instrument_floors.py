"""Stage-2 floor tests (D4) -- the oracle ceiling + MDE + the two-gate floor.

The floor entrypoint ``churn.instrument.floors:compute_floors`` derives every RNG stream from the
beacon randomness (domain-separated tags), so no seed is author-chosen. These tests run at reduced
sizes (a dev beacon string) and check: the oracle ceiling reproduces the frozen regime
(oracle >> static), the floor = max(MDE, 0.5*recoverable_lift) per metric, and both floors land in
the disclosed ~0.08 band (substantive-gate dominated).
"""

from __future__ import annotations

import copy

import pytest

from churn.instrument import floors as F
from churn.simulator import params as P


@pytest.fixture(scope="module")
def small_params() -> dict:
    # Reduced sizes so the test is fast; the real compute_floors uses the frozen n_oracle=200k.
    params = copy.deepcopy(P.load_params())
    params["floor_design"]["n_oracle"] = 30_000
    params["floor_design"]["mde_paired_mc_replicates"] = 400
    return params


DEV_BEACON = "a" * 64  # a valid 64-hex dev randomness; NOT the confirmatory beacon (Stage 2 CI)


@pytest.fixture(scope="module")
def result(small_params):
    return F.compute_floors(small_params, DEV_BEACON, mde_pool=12_000)


def test_oracle_ceiling_reproduces_the_frozen_regime(result):
    roc = result["ceiling"]["roc_auc"]
    assert 0.77 <= roc["oracle"] <= 0.83
    assert 0.60 <= roc["static"] <= 0.68
    assert roc["recoverable_lift"] == pytest.approx(roc["oracle"] - roc["static"], abs=1e-9)
    assert roc["recoverable_lift"] >= 0.12


def test_floor_is_the_max_of_the_two_gates(result):
    fd = result["floor_design"]
    for metric in ("roc_auc", "pr_auc"):
        mde = result["mde"][metric]
        substantive = fd["substantive_fraction_f"] * result["ceiling"][metric]["recoverable_lift"]
        assert result["floors"][metric] == pytest.approx(max(mde, substantive))
        assert result["floors"][metric] > 0


def test_floor_lands_in_the_disclosed_band(result):
    # ~0.5 * recoverable_lift(~0.16) ~ 0.08; both metrics near there.
    assert 0.05 <= result["floors"]["roc_auc"] <= 0.12
    assert 0.05 <= result["floors"]["pr_auc"] <= 0.14


def test_seeds_are_beacon_derived_not_author_chosen(result):
    # The recorded seeds must be the domain-tagged KDF of the beacon randomness (D4).
    from churn.simulator import beacon as B

    assert result["seeds"]["oracle"] == B.derive_seed_hex(DEV_BEACON, B.DOMAIN_TAGS["oracle"])
    assert result["seeds"]["mde"] == B.derive_seed_hex(DEV_BEACON, B.DOMAIN_TAGS["mde"])


def test_determinism_same_beacon_same_floors(small_params):
    a = F.compute_floors(small_params, DEV_BEACON, mde_pool=8_000)
    b = F.compute_floors(small_params, DEV_BEACON, mde_pool=8_000)
    assert a["floors"] == b["floors"]
