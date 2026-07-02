"""Stage-2 numeric floors (D4) -- ``churn.instrument.floors:compute_floors`` (frozen entrypoint).

Per metric (ROC-AUC and PR-AUC) the confirmatory paired-bootstrap CI lower bound must exceed
``floor = max(MDE, f * recoverable_lift)``:

  * Gate 1 (detectability) -- MDE = ``mde_z_multiplier * SE_dAUC``, the empirical paired-MC standard
    error of the static+event vs static ROC/PR lift at the confirmatory ``n = 1600``.
  * Gate 2 (substantive recovery) -- ``f * recoverable_lift``, ``recoverable_lift = oracle_auc -
    static_auc`` (the analytic Bayes hazard oracle vs the frozen analysis static model, at
    ``n_oracle``). ``f = 0.5`` is pre-registered.

**Every RNG stream is derived from the beacon randomness via domain-separated tags** (``oracle``,
``mde``) -- no seed is author-chosen, and ``R`` postdates the analysis freeze (D3/D4), so the floors
cannot be shopped. Part of the ANALYSIS set hashed at Stage 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn import config
from churn.featurestore import offline as OFF
from churn.instrument import model as M
from churn.simulator import beacon as B
from churn.simulator import generate as G
from churn.simulator import kernels as K
from churn.simulator import latent as L
from churn.simulator import params as P

_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
CONFIRMATORY_N = 1600  # the anchor n the MDE + confirmatory bootstrap are defined at (D5.6)


def compute_floors(params: dict, beacon_randomness_hex: str, mde_pool: int = 16_000) -> dict:
    """Oracle ceiling + MDE + the two-gate floor per metric, all beacon-seeded (D4)."""
    fd = params["floor_design"]
    oracle_seed = B.derive_seed_int(beacon_randomness_hex, B.DOMAIN_TAGS["oracle"])
    mde_seed = B.derive_seed_int(beacon_randomness_hex, B.DOMAIN_TAGS["mde"])

    ceiling = _oracle_ceiling(params, oracle_seed, int(fd["n_oracle"]))
    mde = _mde(
        params,
        mde_seed,
        int(fd["mde_paired_mc_replicates"]),
        mde_pool,
        float(fd["mde_z_multiplier"]),
    )

    f = float(fd["substantive_fraction_f"])
    floors = {m: max(mde[m], f * ceiling[m]["recoverable_lift"]) for m in fd["metrics"]}
    return {
        "floors": floors,
        "ceiling": ceiling,
        "mde": mde,
        "floor_design": fd,
        "seeds": {
            "oracle": B.derive_seed_hex(beacon_randomness_hex, B.DOMAIN_TAGS["oracle"]),
            "mde": B.derive_seed_hex(beacon_randomness_hex, B.DOMAIN_TAGS["mde"]),
        },
    }


def _anchor_static() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


def _oracle_ceiling(params: dict, seed_int: int, n: int) -> dict:
    """oracle_auc (analytic hazard) vs static_auc (frozen analysis model) at n_oracle (D4/§8)."""
    anchor = _anchor_static()
    q_anchor = P.quality_index(anchor[G.STATIC_FEATURES], params)
    boot_rng, lat_rng, lab_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed_int).spawn(3)
    )
    idx = boot_rng.integers(0, len(anchor), size=n)
    static_rows = anchor.iloc[idx].reset_index(drop=True)
    q = q_anchor[idx]
    h_t0 = L.health_at_t0(L.simulate_health_paths(q, params, lat_rng))
    haz = params["hazard"]
    oracle_score = K.hazard_prob(h_t0, q, haz["alpha0"], haz["alpha_h"], haz["alpha_stat"])
    y = (lab_rng.random(n) < oracle_score).astype(int)

    half = n // 2
    pipe = M.build_pipeline("static").fit(static_rows.iloc[:half][M.STATIC_FEATURES], y[:half])
    static_proba = pipe.predict_proba(static_rows.iloc[half:][M.STATIC_FEATURES])[:, 1]
    y_hold, oracle_hold = y[half:], oracle_score[half:]

    out = {}
    for metric, fn in _METRICS.items():
        oracle = float(fn(y_hold, oracle_hold))
        static = float(fn(y_hold, static_proba))
        out[metric] = {"oracle": oracle, "static": static, "recoverable_lift": oracle - static}
    return out


def _mde(params: dict, seed_int: int, replicates: int, pool_size: int, z_mult: float) -> dict:
    """MDE = z_mult * empirical SE(dAUC) from paired n=1600 subsamples of a leak-free pool (D4)."""
    pool_seed, cv_seed, sub_seed = np.random.SeedSequence(seed_int).spawn(3)
    pool = G.build_population_dataset(
        params, int(pool_seed.generate_state(1)[0]), n_synthetic=pool_size
    )
    pit = OFF.compute_pit_features(pool.events, pool.cohort)
    design = M.assemble_design(pool, pit)
    cv = int(cv_seed.generate_state(1)[0])
    p_static = M.group_oof_proba("static", design.X, design.y, design.groups, seed=cv)
    p_event = M.group_oof_proba("static_event", design.X, design.y, design.groups, seed=cv)

    rng = np.random.default_rng(sub_seed)
    y, N = design.y, len(design.y)
    diffs = {m: [] for m in _METRICS}
    for _ in range(replicates):
        idx = rng.choice(N, size=min(CONFIRMATORY_N, N), replace=False)
        if np.unique(y[idx]).size < 2:
            continue
        for m, fn in _METRICS.items():
            diffs[m].append(fn(y[idx], p_event[idx]) - fn(y[idx], p_static[idx]))
    return {m: z_mult * float(np.std(diffs[m], ddof=1)) for m in _METRICS}


def clears(delta_ci: dict, floor: float) -> bool:
    """A metric clears iff the paired-bootstrap lower bound strictly exceeds its floor."""
    return delta_ci["diff_lo"] > floor
