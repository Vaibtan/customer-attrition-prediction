"""Confirmatory measurement entrypoint (§4.6 / D2/D3) -- the single-shot, tamper-evident run.

Given the beacon randomness for round ``R`` (fetched by CI after ``R`` emits, keyed to the trusted
merge clock), this: derives the confirmatory data seed + bootstrap seed by domain-separated KDF,
builds the anchor confirmatory dataset (real static, resimulated label), computes the three named
baselines + the paired ΔROC/ΔPR positive control, computes the numeric floors, takes the verdict
(clears iff BOTH lower bounds exceed BOTH floors), and writes a tamper-evident artifact under
``reports/instrument_validation/`` carrying the lock hash, analysis-code hash, git SHA, clean-tree
marker, the beacon-derived seeds, the raw predictions + their content hash, and the verdict.

A confirmatory **miss is a recorded null** (strict stopping rule, D3): no silent re-run. Per D7.1
the world is fixed -- a miss may not trigger another recovery-feasibility re-lock.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from churn import config, registry
from churn.featurestore import offline as OFF
from churn.instrument import experiment as EXP
from churn.instrument import floors as FL
from churn.simulator import beacon as B
from churn.simulator import generate as G
from churn.simulator import lock as LK
from churn.simulator import params as P

RESULTS_DIR = config.ROOT / "reports" / "instrument_validation"


def _clean_tree() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=config.ROOT, text=True)
        return out.strip() == ""
    except (OSError, subprocess.SubprocessError):
        return False


def _predictions_frame(base: dict, dataset) -> pd.DataFrame:
    ids = dataset.labels["customer_id"].to_numpy()
    return (
        pd.DataFrame(
            {
                "customer_id": ids,
                "y": base["_y"],
                "p_static": base["_p_static"],
                "p_event": base["_p_event"],
            }
        )
        .sort_values("customer_id")
        .reset_index(drop=True)
    )


def _predictions_sha256(preds: pd.DataFrame) -> str:
    # Canonical, precision-pinned serialization so CI can regenerate + compare row-by-row (D2).
    rows = [
        f"{r.customer_id}|{int(r.y)}|{r.p_static:.10f}|{r.p_event:.10f}"
        for r in preds.itertuples(index=False)
    ]
    return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


def run_confirmatory(
    beacon_randomness_hex: str,
    out_dir: Path | None = None,
    kind: str = "confirmatory",
    n_rounds: int = 2000,
    floor_kwargs: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Execute the single confirmatory run and write the tamper-evident artifact."""
    params = params if params is not None else P.load_params()
    drifts = LK.check_lock()
    if drifts and kind == "confirmatory":
        raise RuntimeError(f"lock drift -- refusing a confirmatory run: {drifts}")

    conf_seed = B.derive_seed_int(beacon_randomness_hex, B.DOMAIN_TAGS["confirmatory"])
    boot_seed = B.derive_seed_int(beacon_randomness_hex, B.DOMAIN_TAGS["confirmatory_bootstrap"])
    cv_seed = int(np.random.SeedSequence(conf_seed).generate_state(1)[0])

    dataset = G.build_population_dataset(params, conf_seed, n_synthetic=0)
    pit = OFF.compute_pit_features(dataset.events, dataset.cohort)
    base = EXP.compute_baselines(dataset, pit, seed=cv_seed)
    control = {
        "delta_roc": EXP.paired_diff_ci(
            base["_y"],
            base["_p_event"],
            base["_p_static"],
            _roc,
            groups=base["_groups"],
            n_rounds=n_rounds,
            seed=boot_seed,
        ),
        "delta_pr": EXP.paired_diff_ci(
            base["_y"],
            base["_p_event"],
            base["_p_static"],
            _ap,
            groups=base["_groups"],
            n_rounds=n_rounds,
            seed=boot_seed + 1,
        ),
    }
    floors = FL.compute_floors(params, beacon_randomness_hex, **(floor_kwargs or {}))

    verdict = {
        "roc_clears": FL.clears(control["delta_roc"], floors["floors"]["roc_auc"]),
        "pr_clears": FL.clears(control["delta_pr"], floors["floors"]["pr_auc"]),
    }
    verdict["pass"] = bool(verdict["roc_clears"] and verdict["pr_clears"])

    preds = _predictions_frame(base, dataset)
    artifact = {
        "kind": kind,
        "beacon_randomness": beacon_randomness_hex,
        "seeds": {
            "confirmatory": B.derive_seed_hex(beacon_randomness_hex, B.DOMAIN_TAGS["confirmatory"]),
            "confirmatory_bootstrap": B.derive_seed_hex(
                beacon_randomness_hex, B.DOMAIN_TAGS["confirmatory_bootstrap"]
            ),
            "cv_fold_seed": cv_seed,
            **floors["seeds"],
        },
        "baselines": {k: v for k, v in base.items() if not k.startswith("_")},
        "positive_control": control,
        "ceiling": floors["ceiling"],
        "mde": floors["mde"],
        "floors": floors["floors"],
        "verdict": verdict,
        "predictions_sha256": _predictions_sha256(preds),
        "provenance": {
            "lock": LK.load_lock(),
            "analysis_code_sha256": LK.analysis_code_sha256(),
            "git_sha": registry.git_sha(),
            "clean_tree": _clean_tree(),
            "versions": {
                "scikit_learn": __import__("sklearn").__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }

    out_dir = (
        Path(out_dir) if out_dir else (RESULTS_DIR / ("dryrun" if kind != "confirmatory" else ""))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{kind}.json").write_text(
        json.dumps(registry.to_jsonable(artifact), indent=2) + "\n", newline="\n"
    )
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    return artifact


def _roc(y, p):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, p)


def _ap(y, p):
    from sklearn.metrics import average_precision_score

    return average_precision_score(y, p)
