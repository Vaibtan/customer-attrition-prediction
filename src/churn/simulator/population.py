"""Population sampler (D6.4 / D5.7) -- the ~50k synthetic distribution anchor.

Builds a production-shaped synthetic population whose static attributes match the real 1,600
(``data/customer_data.csv``): the plan marginal, the plan-conditional distributions of
``monthly_spend`` / ``account_age_days`` / ``region`` (the three fitted joints), and the global
``device_type`` / ``avg_order_value`` marginals. Per D5.7 the non-anchored columns are drawn
*conditionally independent given plan*, so each is bootstrapped independently within its plan
stratum (exactly preserving each ``plan x column`` joint without imposing a parametric form on the
heavy-tailed spend/aov columns). Light jitter on the continuous numerics stops the 50k from being
the 1,600 tiled ~31x.

The 1,600 anchor rows keep their **real** static attributes verbatim (only their health, events, and
label are re-simulated downstream); their real churn label is carried on ``real_churned`` and feeds
**only** ``real_static_reference_auc`` (never the event experiment). No label information enters the
synthetic draw (D6 invariant i).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config
from churn.simulator import params as P
from churn.simulator import rng as R

STATIC_COLS = ["region", "device_type", "subscription_plan", *P.STATIC_NUMERICS]
# Multiplicative log-normal jitter sigma. Multiplicative (not additive) because these columns are
# non-negative and heavy-tailed (monthly_spend: median 56 vs mean 243) -- additive jitter scaled to
# the tail-dominated std would swamp typical values and, after clipping negatives at 0, bias the
# median upward. exp(N(0, sigma)) preserves positivity + the median while de-duplicating tiles.
_JITTER_SIGMA = 0.05


def _valid_range(col: str) -> tuple[float, float]:
    lo, hi = config.VALID_RANGES.get(col, (None, None))
    return (0.0 if lo is None else float(lo), np.inf if hi is None else float(hi))


def _bootstrap_column(
    pool: np.ndarray, k: int, rng: np.random.Generator, *, jitter: bool, col: str
) -> np.ndarray:
    """Draw ``k`` values by bootstrap-resampling ``pool`` (NaN-imputed), optionally jittered."""
    idx = rng.integers(0, len(pool), size=k)
    out = pool[idx].astype(np.float64)
    if jitter:
        out = out * np.exp(rng.normal(0.0, _JITTER_SIGMA, size=k))
    lo, hi = _valid_range(col)
    return np.clip(out, lo, hi)


def _impute_pool(values: pd.Series) -> np.ndarray:
    """A NaN-free sampling pool: impute the column median (frozen-median-consistent with q(x))."""
    return values.fillna(values.median()).to_numpy(dtype=np.float64)


def sample_population(params: dict, seed: int, n_synthetic: int | None = None) -> pd.DataFrame:
    """Anchor (real static, verbatim) + ``n_synthetic`` synthetic customers; seed-deterministic."""
    anchor = pd.read_csv(config.DATA_PATH)
    anchor_n = int(params["population"]["anchor_n"])
    if len(anchor) != anchor_n:
        raise ValueError(f"anchor has {len(anchor)} rows, params expects {anchor_n}")
    if n_synthetic is None:
        n_synthetic = int(params["population"]["size"]) - anchor_n

    rng = R.make_streams(seed)["population"]

    anchor_block = anchor[["customer_id", *STATIC_COLS]].copy()
    anchor_block["customer_id"] = anchor_block["customer_id"].astype(str)
    anchor_block["is_anchor"] = True
    anchor_block["real_churned"] = anchor[config.TARGET].astype("Int64")

    syn = _sample_synthetic(anchor, n_synthetic, rng)
    syn["is_anchor"] = False
    syn["real_churned"] = pd.array([pd.NA] * n_synthetic, dtype="Int64")

    cols = ["customer_id", *STATIC_COLS, "is_anchor", "real_churned"]
    return pd.concat([anchor_block[cols], syn[cols]], ignore_index=True)


def _sample_synthetic(anchor: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    plans = sorted(anchor["subscription_plan"].unique())
    counts = anchor["subscription_plan"].value_counts().reindex(plans).to_numpy(dtype=float)
    probs = counts / counts.sum()
    syn_plan = rng.choice(plans, size=n, p=probs)

    out: dict[str, np.ndarray] = {
        "subscription_plan": syn_plan,
        "region": np.empty(n, dtype=object),
        "monthly_spend": np.empty(n, dtype=np.float64),
        "account_age_days": np.empty(n, dtype=np.float64),
        "avg_order_value": np.empty(n, dtype=np.float64),
    }
    # Plan-conditional, independent bootstrap for the anchored columns (D5.7). avg_order_value is
    # drawn plan-conditionally too (D6.4, Codex round): it feeds q(x) with weight 0.15, and the
    # frozen regime was tuned via a whole-anchor-row q bootstrap that preserves its real dependence
    # on plan -- so a global-marginal draw would flatten P(aov|plan) and mis-scale the Stage-2 floor
    # MC relative to the anchor the confirmatory control is measured on.
    for plan in plans:
        mask = syn_plan == plan
        k = int(mask.sum())
        if k == 0:
            continue
        grp = anchor[anchor["subscription_plan"] == plan]
        out["region"][mask] = _bootstrap_categorical(grp["region"], k, rng)
        out["monthly_spend"][mask] = _bootstrap_column(
            _impute_pool(grp["monthly_spend"]), k, rng, jitter=True, col="monthly_spend"
        )
        out["account_age_days"][mask] = _bootstrap_column(
            _impute_pool(grp["account_age_days"]), k, rng, jitter=True, col="account_age_days"
        )
        out["avg_order_value"][mask] = _bootstrap_column(
            _impute_pool(grp["avg_order_value"]), k, rng, jitter=True, col="avg_order_value"
        )
    # device_type carries zero q-weight and is in no fitted joint: a global marginal is correct.
    out["device_type"] = _bootstrap_categorical(anchor["device_type"], n, rng)
    out["account_age_days"] = np.round(out["account_age_days"]).astype(np.int64)
    out["customer_id"] = np.array([f"SYN{i:06d}" for i in range(n)], dtype=object)
    return pd.DataFrame(out)


def _bootstrap_categorical(values: pd.Series, k: int, rng: np.random.Generator) -> np.ndarray:
    pool = values.to_numpy(dtype=object)
    return pool[rng.integers(0, len(pool), size=k)]
