"""Cohort diagnostics (§4.5) -- anchor vs synthetic, side by side, before any blended claim.

Reports base rate, categorical marginals, numeric summaries + missingness, and static-only AUC for
the anchor (real static) and synthetic cohorts. These **scope** a result to a domain (they say how
alike the two populations are); they are explicitly **not** a validity guarantee (R2 verdict 6).
Descriptive only -- not part of the confirmatory prediction path, so not in the analysis-code hash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.instrument import model as M

_CATEGORICAL = M.STATIC_CATEGORICAL
_NUMERIC = M.STATIC_NUMERIC


def _numeric_summary(df: pd.DataFrame) -> dict:
    out = {}
    for col in _NUMERIC:
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=0)),
            "missing_frac": float(s.isna().mean()),
        }
    return out


def _categorical_marginals(df: pd.DataFrame) -> dict:
    return {col: df[col].value_counts(normalize=True).round(4).to_dict() for col in _CATEGORICAL}


def _static_auc(dataset, cohort_ids: np.ndarray, pit_features: pd.DataFrame, seed: int) -> float:
    sub = dataset.customers[dataset.customers["customer_id"].isin(cohort_ids)]
    static = sub[["customer_id", *M.STATIC_FEATURES]].merge(pit_features, on="customer_id")
    y = dataset.labels.set_index("customer_id").loc[static["customer_id"], "y"].to_numpy(int)
    groups = static["customer_id"].to_numpy()
    proba = M.group_oof_proba(
        "static", static[M.STATIC_FEATURES + M.EVENT_FEATURES], y, groups, seed
    )
    return M.auc(y, proba)


def _cohort_block(dataset, ids: np.ndarray, pit_features: pd.DataFrame, seed: int) -> dict:
    customers = dataset.customers[dataset.customers["customer_id"].isin(ids)]
    labels = dataset.labels[dataset.labels["customer_id"].isin(ids)]
    return {
        "n": int(len(customers)),
        "base_rate": float(labels["y"].mean()),
        "categorical_marginals": _categorical_marginals(customers),
        "numeric_summary": _numeric_summary(customers),
        "static_auc": _static_auc(dataset, ids, pit_features, seed),
    }


def cohort_diagnostics(
    dataset, pit_features: pd.DataFrame, seed: int = 42, synthetic_cap: int = 8000
) -> dict:
    """Side-by-side anchor vs synthetic diagnostics; the synthetic AUC uses a capped subsample."""
    anchor_ids = dataset.customers.loc[dataset.customers["is_anchor"], "customer_id"].to_numpy()
    syn_ids = dataset.customers.loc[~dataset.customers["is_anchor"], "customer_id"].to_numpy()
    if len(syn_ids) > synthetic_cap:
        syn_ids = np.random.default_rng(seed).choice(syn_ids, size=synthetic_cap, replace=False)
    return {
        "anchor": _cohort_block(dataset, anchor_ids, pit_features, seed),
        "synthetic": _cohort_block(dataset, syn_ids, pit_features, seed),
        "note": (
            "Cohort diagnostics SCOPE a synthetic-domain result to its population; they are not a "
            "guarantee of external validity. The anchor keeps real static (real missingness too); "
            "the synthetic cohort is complete by construction (D6.4)."
        ),
    }
