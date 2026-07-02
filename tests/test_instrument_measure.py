"""Confirmatory measurement tests (§4.6 / D2) -- the tamper-evident single-shot machinery.

Runs the entrypoint at reduced sizes with a dev beacon string (kind='dry_run', so the lock-drift
refusal + RESULTS separation don't apply). Checks the artifact is structurally complete and
tamper-evident (lock + code hash + git SHA + predictions hash embedded), reproducible, and that the
positive control clears both floors on an exploratory-equivalent draw.
"""

from __future__ import annotations

import copy

import pytest

from churn.instrument import measure as MEAS
from churn.simulator import params as P

DEV_BEACON = "b" * 64


@pytest.fixture(scope="module")
def small_params() -> dict:
    params = copy.deepcopy(P.load_params())
    params["floor_design"]["n_oracle"] = 30_000
    params["floor_design"]["mde_paired_mc_replicates"] = 300
    return params


@pytest.fixture(scope="module")
def artifact(tmp_path_factory, small_params):
    out = tmp_path_factory.mktemp("iv")
    return MEAS.run_confirmatory(
        DEV_BEACON,
        out_dir=out,
        kind="dry_run",
        n_rounds=500,
        floor_kwargs={"mde_pool": 8000},
        params=small_params,
    )


def test_artifact_is_structurally_complete(artifact):
    assert artifact["kind"] == "dry_run"
    for key in (
        "real_static_reference_auc",
        "synthetic_static_auc",
        "synthetic_static_plus_event_auc",
    ):
        assert key in artifact["baselines"]
    assert set(artifact["floors"]) == {"roc_auc", "pr_auc"}
    assert "roc_clears" in artifact["verdict"] and "pass" in artifact["verdict"]


def test_provenance_is_tamper_evident(artifact):
    prov = artifact["provenance"]
    assert prov["analysis_code_sha256"].startswith("sha256:")
    assert prov["lock"]["hashes"]  # the frozen lock is embedded
    assert "git_sha" in prov and "clean_tree" in prov
    assert artifact["predictions_sha256"].startswith("sha256:")
    # Seeds are beacon-derived, not stored as free values.
    from churn.simulator import beacon as B

    assert artifact["seeds"]["confirmatory"] == B.derive_seed_hex(
        DEV_BEACON, B.DOMAIN_TAGS["confirmatory"]
    )


def test_positive_control_clears_both_floors(artifact):
    # On this (dev-beacon) draw the leak-free instrument recovers the injected signal.
    assert artifact["positive_control"]["delta_roc"]["diff_lo"] > artifact["floors"]["roc_auc"]
    assert artifact["positive_control"]["delta_pr"]["diff_lo"] > artifact["floors"]["pr_auc"]
    assert artifact["verdict"]["pass"] is True


def test_determinism_same_beacon_same_result(tmp_path, small_params):
    a = MEAS.run_confirmatory(
        DEV_BEACON,
        out_dir=tmp_path / "a",
        kind="dry_run",
        n_rounds=300,
        floor_kwargs={"mde_pool": 6000},
        params=small_params,
    )
    b = MEAS.run_confirmatory(
        DEV_BEACON,
        out_dir=tmp_path / "b",
        kind="dry_run",
        n_rounds=300,
        floor_kwargs={"mde_pool": 6000},
        params=small_params,
    )
    assert a["predictions_sha256"] == b["predictions_sha256"]
    assert a["verdict"] == b["verdict"]
