"""Loader + binding for the frozen simulator params (single source of truth, like config.py).

``simulator.params.json`` holds every Stage-1 world constant (D5); this module loads it and
binds the frozen static-encoding constants to the pure kernels so downstream code (the sanity
test, and the Phase-1 generator) computes the customer-quality index ``q(x)`` the *one* declared
way. There is deliberately no confirmatory seed here -- that is beacon-derived at Stage 2 (D3).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churn import config
from churn.simulator import kernels as K

SIM_PARAMS_PATH = config.ROOT / "simulator.params.json"

# The static numerics that feed the quality index (region/device_type carry zero weight, D5.4).
STATIC_NUMERICS = ["account_age_days", "monthly_spend", "avg_order_value"]

REQUIRED_TOP_KEYS = {
    "schema_version",
    "latent",
    "quality_index",
    "hazard",
    "events",
    "temporal",
    "population",
    "floor_design",
    "controls",
    "tuning",
}


def load_params(path=SIM_PARAMS_PATH) -> dict:
    """Load + minimally validate the frozen params. Raises on missing/renamed top-level keys."""
    params = json.loads(path.read_text())
    missing = REQUIRED_TOP_KEYS - params.keys()
    if missing:
        raise ValueError(f"simulator.params.json missing keys: {sorted(missing)}")
    if params["schema_version"] != 1:
        raise ValueError(f"unsupported simulator params schema_version: {params['schema_version']}")
    return params


def q_raw(static_df: pd.DataFrame, plan_map: dict, weights: dict, numeric_standardize: dict):
    """Raw (unstandardized) customer-quality index from explicit encoding constants (D5.4).

    The single q-encoder shared by the tuning harness (which passes freshly-computed anchor stats
    it is about to freeze) and by ``quality_index`` (which passes the frozen stats) -- so the
    stats-computing and stats-applying paths cannot silently diverge.
    """
    plan = static_df["subscription_plan"].map(plan_map).to_numpy(dtype=float)
    z = {}
    for col in STATIC_NUMERICS:
        ns = numeric_standardize[col]
        s = pd.to_numeric(static_df[col], errors="coerce").fillna(ns["median"])
        z[col] = K.standardize(s.to_numpy(dtype=float), ns["mean"], ns["std"])
    return K.quality_index_raw(
        plan, z["account_age_days"], z["monthly_spend"], z["avg_order_value"], weights
    )


def quality_index(static_df: pd.DataFrame, params: dict) -> np.ndarray:
    """Standardized customer-quality index q(x) using the FROZEN encoding (D5.4).

    Reads the plan map, direction weights, per-numeric standardization, and the final q-raw
    standardization straight from ``params`` -- never recomputed from data -- so every caller
    gets the identical frozen index.
    """
    qi = params["quality_index"]
    raw = q_raw(static_df, qi["plan_score_map"], qi["weights"], qi["numeric_standardize"])
    qs = qi["q_raw_standardize"]
    return (raw - qs["mean"]) / qs["std"]
