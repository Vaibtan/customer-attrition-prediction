"""simulator.lock.json -- the Stage-1 hash + beacon lock, and check_lock (D1/D2).

``check_lock`` recomputes the sha256 of every FROZEN file and asserts equality with the committed
lock; it exits non-zero on drift. It is reused by the Layer-1 CI job and (at Stage 2) asserted at
runtime by the measurement entrypoint. Regenerate the lock with:

    python -m churn.simulator.lock --write

Hashes are computed over **line-ending-normalized** bytes (CRLF/CR -> LF) so the lock is stable
across the Windows dev box and Linux CI -- a git autocrlf checkout must not break the freeze.

Stage-1 FROZEN set = {docs/simulator_spec.md, simulator.params.json}. The analysis_spec hash, the
analysis-code hash, and the numeric floor values are added at Stage 2 (end of Phase 1).
"""

from __future__ import annotations

import hashlib
import json
import sys

from churn import config, registry
from churn.simulator import beacon
from churn.simulator import params as P

SIM_LOCK_PATH = config.ROOT / "simulator.lock.json"

# FROZEN text set, hashed with line-ending normalization. kernels.py is hashed too (the form-
# computing code): the golden vectors pin that it computes the declared forms, but freezing the
# code itself closes the hole where a formula + its golden expected value are co-edited undetected.
HASHED_FILES = [
    "docs/simulator_spec.md",
    "simulator.params.json",
    # The world-construction + derivation code (form kernels, the q(x) encoder, and the beacon
    # round/KDF derivation). Freezing these closes the holes where a formula/encoder/KDF and its
    # golden expected value or recipe description could be co-edited with no lock drift.
    "src/churn/simulator/kernels.py",
    "src/churn/simulator/params.py",
    "src/churn/simulator/beacon.py",
]
# The distribution anchor is DATA (not text): hashed raw (no normalization), pinned separately,
# because the frozen standardization stats in simulator.params.json are derived from it.
ANCHOR_REL = "data/customer_data.csv"
LOCK_VERSION = 1


def _normalized_sha256(rel_path: str) -> str:
    raw = (config.ROOT / rel_path).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return "sha256:" + hashlib.sha256(normalized).hexdigest()


def compute_hashes() -> dict:
    return {rel: _normalized_sha256(rel) for rel in HASHED_FILES}


def load_lock(path=SIM_LOCK_PATH) -> dict:
    return json.loads(path.read_text())


def build_lock() -> dict:
    prm = P.load_params()
    return {
        "lock_version": LOCK_VERSION,
        "stage": 1,
        "description": (
            "Stage-1 pre-registration lock (PHASE0_LOCK_DECISIONS.md D1-D5). Freezes the world "
            "(docs/simulator_spec.md + simulator.params.json) and the confirmatory-seed / "
            "floor-RNG beacon recipe. The analysis spec, analysis-code hash, and numeric floors "
            "are Stage 2."
        ),
        "git_sha_at_generation": registry.git_sha(),
        "git_sha_note": (
            "HEAD when the lock was written (typically the PARENT of the lock commit); "
            "informational provenance only -- NOT verified by check_lock and not dirty-checked."
        ),
        "hashed_files": HASHED_FILES,
        "hash_normalization": "crlf-and-cr-to-lf-before-sha256 (text members only)",
        "hashes": compute_hashes(),
        "anchor": {
            "path": ANCHOR_REL,
            "sha256": registry.data_sha256(),
            "note": "raw sha256 of the anchor (no normalization); verified by check_lock.",
        },
        "beacon": beacon.recipe(),
        "controls": prm["controls"],
        "stage2": {
            "analysis_spec_sha256": None,
            "analysis_code_sha256": None,
            "numeric_floors": None,
            "note": (
                "Added at Stage 2 (end of Phase 1): analysis_spec.json hash + analysis-code hash + "
                "numeric floor values computed from the frozen pipeline after beacon round R emits."
            ),
        },
    }


def write_lock(path=SIM_LOCK_PATH) -> dict:
    lock = build_lock()
    path.write_text(json.dumps(registry.to_jsonable(lock), indent=2) + "\n", newline="\n")
    return lock


def check_lock(path=SIM_LOCK_PATH) -> list[str]:
    """Return a list of drift messages (empty list == the lock is consistent)."""
    lock = load_lock(path)
    current = compute_hashes()
    locked = lock.get("hashes", {})
    drifts: list[str] = []
    for rel in HASHED_FILES:
        if rel not in locked:
            drifts.append(f"{rel}: missing from lock hashes")
        elif locked[rel] != current[rel]:
            drifts.append(f"{rel}: lock {locked[rel]} != current {current[rel]}")
    if lock.get("hashed_files") != HASHED_FILES:
        drifts.append("hashed_files: lock list != code HASHED_FILES")
    # The anchor data is hashed raw (it is data, not normalized text).
    if lock.get("anchor", {}).get("sha256") != registry.data_sha256():
        drifts.append(f"{ANCHOR_REL}: anchor sha256 != current")
    # The committed beacon recipe + the controls copy cannot be silently edited in the JSON.
    if lock.get("beacon") != beacon.recipe():
        drifts.append("beacon: committed recipe != churn.simulator.beacon.recipe()")
    if lock.get("controls") != P.load_params()["controls"]:
        drifts.append("controls: lock copy != simulator.params.json controls")
    return drifts


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--write" in argv:
        write_lock()
        print(f"Wrote {SIM_LOCK_PATH.name}")
        return 0
    drifts = check_lock()
    if drifts:
        print("LOCK DRIFT DETECTED:")
        for d in drifts:
            print("  - " + d)
        return 1
    print("simulator.lock.json consistent: " + ", ".join(HASHED_FILES) + " + beacon recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
