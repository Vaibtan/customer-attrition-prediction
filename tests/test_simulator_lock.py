"""Lock-consistency tests (D1/D2): check_lock passes clean, catches drift, and round-trips.

The committed simulator.lock.json must agree with the frozen files + the beacon recipe; a change
to either without a deliberate `--write` regeneration must fail check_lock (the Layer-1 CI guard).
"""

from __future__ import annotations

import json

from churn.simulator import beacon
from churn.simulator import lock as L


def test_committed_lock_is_consistent():
    assert L.check_lock() == []


def test_compute_hashes_match_the_committed_lock():
    committed = L.load_lock()["hashes"]
    assert L.compute_hashes() == committed
    assert all(v.startswith("sha256:") for v in committed.values())


def test_committed_beacon_recipe_matches_code():
    assert L.load_lock()["beacon"] == beacon.recipe()


def test_check_lock_detects_a_tampered_hash(tmp_path):
    tampered = L.load_lock()
    tampered["hashes"]["simulator.params.json"] = "sha256:" + "0" * 64
    p = tmp_path / "simulator.lock.json"
    p.write_text(json.dumps(tampered), newline="\n")
    drifts = L.check_lock(p)
    assert any("simulator.params.json" in d for d in drifts)


def test_check_lock_detects_a_tampered_beacon(tmp_path):
    tampered = L.load_lock()
    tampered["beacon"]["chain_hash"] = "deadbeef"
    p = tmp_path / "simulator.lock.json"
    p.write_text(json.dumps(tampered), newline="\n")
    drifts = L.check_lock(p)
    assert any("beacon" in d for d in drifts)


def test_check_lock_flags_a_missing_hash_entry(tmp_path):
    tampered = L.load_lock()
    del tampered["hashes"]["docs/simulator_spec.md"]
    p = tmp_path / "simulator.lock.json"
    p.write_text(json.dumps(tampered), newline="\n")
    drifts = L.check_lock(p)
    assert any("missing from lock" in d for d in drifts)


def test_world_construction_code_is_in_the_frozen_hash_set():
    # The form kernels, the q(x) encoder, and the beacon derivation code are all hashed, closing
    # the co-edit holes (formula+golden, encoder, KDF-swap) a recipe-dict-only check would miss.
    hashes = L.load_lock()["hashes"]
    for mod in (
        "src/churn/simulator/kernels.py",
        "src/churn/simulator/params.py",
        "src/churn/simulator/beacon.py",
    ):
        assert mod in L.HASHED_FILES
        assert mod in hashes


def test_check_lock_detects_a_tampered_anchor_hash(tmp_path):
    tampered = L.load_lock()
    tampered["anchor"]["sha256"] = "0" * 64
    p = tmp_path / "simulator.lock.json"
    p.write_text(json.dumps(tampered), newline="\n")
    assert any("anchor" in d for d in L.check_lock(p))


def test_check_lock_detects_tampered_controls(tmp_path):
    tampered = L.load_lock()
    tampered["controls"]["label_shuffle"]["method"] = "hand-picked-permutation"
    p = tmp_path / "simulator.lock.json"
    p.write_text(json.dumps(tampered), newline="\n")
    assert any("controls" in d for d in L.check_lock(p))


def test_write_lock_round_trips(tmp_path):
    p = tmp_path / "simulator.lock.json"
    written = L.write_lock(p)
    assert written["hashes"] == L.compute_hashes()
    assert L.check_lock(p) == []


def test_main_returns_zero_when_consistent(capsys):
    assert L.main([]) == 0
    assert "consistent" in capsys.readouterr().out


def test_main_returns_one_and_reports_on_drift(monkeypatch, capsys):
    monkeypatch.setattr(L, "check_lock", lambda *a, **k: ["simulator.params.json: drift"])
    assert L.main([]) == 1
    assert "DRIFT" in capsys.readouterr().out


def test_main_write_flag_regenerates(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(L, "write_lock", lambda *a, **k: calls.setdefault("wrote", True))
    assert L.main(["--write"]) == 0
    assert calls.get("wrote") and "Wrote" in capsys.readouterr().out
