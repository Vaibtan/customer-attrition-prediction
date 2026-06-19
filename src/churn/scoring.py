"""Batch scoring + the stable risk-tier contract.

Risk tiers use cutpoints frozen at training time (persisted in the registry), so
a customer's tier depends only on their own probability — never on who else is
in the batch and never differs between the batch CLI and the API. This is the
fix for Codex finding #4 and is enforced by ``tests/test_scoring_contract.py``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, registry
from .data import load_data, validate_schema
from .interpret import reason_codes_for_frame


def score_to_tier(proba, t_star: float, t_mid: float):
    """Map probabilities to low/medium/high using FIXED cutpoints.

    Vectorised; accepts a scalar or array. Pure function of (proba, cutpoints).
    """
    p = np.asarray(proba, dtype="float64")
    tiers = np.where(p >= t_star, "high", np.where(p >= t_mid, "medium", "low"))
    return tiers.item() if np.isscalar(proba) or p.ndim == 0 else tiers


def score_frame(
    df_raw: pd.DataFrame,
    model,
    t_star: float,
    t_mid: float,
    base_linear=None,
    k: int = 3,
) -> pd.DataFrame:
    """Score a raw customer dataframe → id, probability, tier, (reason codes)."""
    ids = (
        df_raw[config.ID_COL].to_numpy()
        if config.ID_COL in df_raw.columns
        else np.arange(len(df_raw))
    )
    drop = [c for c in (config.ID_COL, config.TARGET) if c in df_raw.columns]
    X = df_raw.drop(columns=drop)

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            config.ID_COL: ids,
            "churn_probability": proba,
            "risk_tier": score_to_tier(proba, t_star, t_mid),
        }
    )
    if base_linear is not None:
        out["top_reason_codes"] = reason_codes_for_frame(base_linear, X, k=k)
    return out


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Batch churn scoring.")
    parser.add_argument("--in", dest="in_path", default=str(config.DATA_PATH))
    parser.add_argument("--out", dest="out_path", default="scored.csv")
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    args = parser.parse_args(argv)

    model, meta, base = registry.load_run(args.run_dir)
    df = load_data(args.in_path)
    validate_schema(df, require_target=False)

    cut = meta["tier_cutpoints"]
    scored = score_frame(df, model, cut["t_star"], cut["t_mid"], base_linear=base)
    scored.to_csv(args.out_path, index=False)
    print(f"Scored {len(scored):,} customers -> {args.out_path}")
    print(scored["risk_tier"].value_counts().rename("count").to_string())


if __name__ == "__main__":
    _cli()
