"""Batch scoring plus the stable risk-tier contract (cutpoints frozen at training)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, registry
from .data import load_data, validate_schema
from .interpret import reason_codes_for_frame


def score_to_tier(proba, t_star: float, t_mid: float):
    """Map probabilities to low/medium/high using fixed cutpoints (scalar or array)."""
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
    """Score a raw customer dataframe into id, probability, tier, (reason codes)."""
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


@dataclass(frozen=True)
class ScoredCustomer:
    """One scored customer as a typed record; the single map from score_frame's columns.

    Consumers (the API envelope) read named attributes, so a column rename in
    score_frame fails here at one seam instead of silently in the response mapping.
    """

    customer_id: object
    churn_probability: float
    risk_tier: str
    top_reason_codes: str | None = None

    @classmethod
    def from_frame(cls, scored: pd.DataFrame) -> list[ScoredCustomer]:
        has_reasons = "top_reason_codes" in scored.columns
        return [
            cls(
                customer_id=row[config.ID_COL],
                churn_probability=float(row["churn_probability"]),
                risk_tier=str(row["risk_tier"]),
                top_reason_codes=row["top_reason_codes"] if has_reasons else None,
            )
            for _, row in scored.iterrows()
        ]


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Batch churn scoring.")
    parser.add_argument("--in", dest="in_path", default=str(config.DATA_PATH))
    parser.add_argument("--out", dest="out_path", default="scored.csv")
    parser.add_argument("--run-dir", dest="run_dir", default=None)
    args = parser.parse_args(argv)

    loaded = registry.load_run(args.run_dir)
    df = load_data(args.in_path)
    validate_schema(df, require_target=False)

    scored = loaded.score(df)
    scored.to_csv(args.out_path, index=False)
    print(f"Scored {len(scored):,} customers -> {args.out_path}")
    print(scored["risk_tier"].value_counts().rename("count").to_string())


if __name__ == "__main__":
    cli()
