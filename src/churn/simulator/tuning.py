"""Scalar Monte-Carlo tuning harness (PHASE0_LOCK_DECISIONS.md D5.9).

Sets the frozen *latent + hazard* coefficients so the frozen world is non-degenerate:
base rate ~= 0.48, static_auc ~= 0.64 (anchor band), oracle_auc ~= 0.80 (real recoverable
headroom). It does **no event generation** -- oracle_auc/static_auc depend only on the
latent + hazard + static design, not on the event-emission links -- so a ~scalar MC suffices.

This is legitimate "target-aware simulator tuning guarded by the lock" (ENHANCEMENT_PLAN.md
§4.7): the world's signal is tuned here, then *frozen*; it never touches the Phase-1 analysis
pipeline or the (beacon-derived) confirmatory seed, so pre-registration integrity holds.

Run ``python -m churn.simulator.tuning`` to (re)generate ``simulator.params.json`` and the
``reports/simulator_tuning.json`` provenance record. The event-emission coefficients (D5.5)
are set a-priori here and are NOT tuned.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn import config, registry
from churn.simulator import kernels as K
from churn.simulator import params as P

SIM_PARAMS_PATH = P.SIM_PARAMS_PATH  # single source for the frozen-params path (params.py)
TUNING_REPORT_PATH = config.ROOT / "reports" / "simulator_tuning.json"

# --- Frozen static-encoding design (D5.3 / D5.4) ---------------------------------------------
PLAN_SCORE_MAP = {"Free": 0.0, "Basic": 1.0, "Premium": 2.0, "Enterprise": 3.0}
QI_WEIGHTS = {"plan": 1.0, "age": 0.25, "spend": 0.20, "aov": 0.15}  # plan dominates (anchor)
STATIC_NUMERICS = P.STATIC_NUMERICS

# --- Fixed latent knobs (D5.1 / D5.9) --------------------------------------------------------
KAPPA = 0.2  # weekly mean-reversion speed
SIGMA_H = 1.0  # stochastic health std at stationarity
DIRECT_STATIC_FRACTION = 0.5  # a_stat = f * A; the rest of static's effect flows via mu(x)

# --- a-priori event-emission coefficients (D5.5) -- NOT tuned; do not affect the ceiling -----
# Health h has ~unit scale (q unit-variance, sigma_h = 1), so h ranges roughly [-3, 3].
EVENT_PARAMS = {
    # Login/engagement family: healthier -> more logins, deeper sessions, more orders (b > 0).
    "login_rate": {"a": float(np.log(4.0)), "b": 0.40},  # ~4 logins/wk at h=0
    "session_depth": {"a": 8.0, "b": 1.00, "tau": 2.0},  # ~8 pages/session at h=0
    "order_rate": {"a": float(np.log(0.5)), "b": 0.30},  # ~0.5 orders/wk at h=0
    # Payment family: unhealthy -> more failures (b < 0).
    "payment_fail": {"a": float(np.log(0.05 / 0.95)), "b": -0.70},  # ~5% failure at h=0
    # Support family: unhealthy -> more tickets, more negative sentiment.
    "support_rate": {"a": float(np.log(0.15)), "b": -0.40},  # ~0.15 tickets/wk at h=0
    "sentiment": {"a": 0.0, "b": 0.50, "tau": 0.5},  # neutral sentiment at h=0
    # Downgrade family: healthier -> rarely downgrades (b < 0).
    "downgrade": {"a": float(np.log(0.02 / 0.98)), "b": -0.60},  # ~2% downgrade/cycle at h=0
}

# --- Tuning targets (D5.9) -------------------------------------------------------------------
TARGET_BASE_RATE = 0.48
TARGET_STATIC_AUC = 0.64
TARGET_ORACLE_AUC = 0.80

# --- Frozen floor-computation DESIGN (D4) ----------------------------------------------------
# Stage-1-fixed so the pass/fail *bar computation* cannot be shopped in Phase 1. Only the RNG
# SEEDS and the numeric floor VALUES are Stage 2 (beacon-derived after round R emits). Recorded
# in the hashed params.json so the spec's "Stage-1-fixed" claim is actually attested by the lock.
FLOOR_DESIGN = {
    "metrics": ["roc_auc", "pr_auc"],
    "n_oracle": 200_000,
    "oracle_score": "analytic_bayes_hazard",  # true P(y|h(t0),q); no fitted estimator (D4)
    "mde_alpha_one_sided": 0.05,
    "mde_power": 0.80,
    "mde_z_multiplier": 2.49,  # z_{1-alpha} + z_power at alpha=0.05, power=0.80
    "mde_paired_mc_replicates": 2000,
    "substantive_fraction_f": 0.5,  # floor = max(MDE, f * recoverable_lift)
    "bootstrap": "percentile_customer_level",
    "floor_entrypoint": "churn.instrument.floors:compute_floors",  # Stage-2 module identity
    "rng_tags": ["oracle", "mde", "floor_bootstrap", "confirmatory_bootstrap"],  # beacon.py domains
}

# Reproducible *development* seed for the tuning MC. This is NOT the confirmatory seed
# (that is beacon-derived at Stage 2 and never committed in the clear -- D3).
DEV_SEED = 20260701
SEARCH_N = 60_000
REPORT_N = 300_000


# --- Anchor static encoding ------------------------------------------------------------------


def load_anchor_static() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    cols = ["region", "device_type", "subscription_plan", *STATIC_NUMERICS]
    return df[cols].copy()


def numeric_stats(static: pd.DataFrame) -> dict:
    """Frozen median (for imputation) + mean/std (for standardization) per static numeric."""
    stats = {}
    for col in STATIC_NUMERICS:
        s = pd.to_numeric(static[col], errors="coerce")
        median = float(s.median())
        filled = s.fillna(median)
        stats[col] = {
            "median": median,
            "mean": float(filled.mean()),
            "std": float(filled.std(ddof=0)),
        }
    return stats


def raw_quality(static: pd.DataFrame, num_stats: dict) -> np.ndarray:
    """q_raw(x) over the anchor (before final standardization), via the shared encoder."""
    return P.q_raw(static, PLAN_SCORE_MAP, QI_WEIGHTS, num_stats)


# --- Scalar Monte-Carlo ----------------------------------------------------------------------


def _draw(q_anchor: np.ndarray, n: int, rng: np.random.Generator):
    """Bootstrap n customers' q from the anchor; draw the health residual eta and label uniforms."""
    idx = rng.integers(0, len(q_anchor), size=n)
    q = q_anchor[idx]
    eta = rng.standard_normal(n)
    u = rng.random(n)
    return q, eta, u


def _aucs(A: float, B: float, alpha0: float, q, eta, u):
    """base_rate, static_auc (q-only), oracle_auc (true prob) for logit = alpha0 - A*q - B*eta."""
    p = K.sigmoid(alpha0 - A * q - B * eta)
    y = (u < p).astype(int)
    base = float(y.mean())
    if base in (0.0, 1.0):
        return base, float("nan"), float("nan")
    # Higher q -> lower churn, so the static churn-score is -q (rank-equivalent to the Bayes
    # static predictor E[y|q] and to a logistic fit on the static features).
    static_auc = float(roc_auc_score(y, -q))
    oracle_auc = float(roc_auc_score(y, p))
    return base, static_auc, oracle_auc


def _bisect(f, lo: float, hi: float, target: float, iters: int = 30) -> float:
    """Bisection for a monotone-increasing f to f(x) == target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_coefficients(q_anchor: np.ndarray, rng: np.random.Generator) -> dict:
    """Find (A, B, alpha0) hitting the AUC + base-rate targets, then back out world coefficients."""
    q, eta, u = _draw(q_anchor, SEARCH_N, rng)

    def A_for_static(B: float) -> float:
        # static_auc increases in A at fixed B.
        return _bisect(lambda A: _aucs(A, B, 0.0, q, eta, u)[1], 0.05, 5.0, TARGET_STATIC_AUC, 28)

    # oracle_auc increases in B once A is re-solved to hold static at target.
    B = _bisect(
        lambda B: _aucs(A_for_static(B), B, 0.0, q, eta, u)[2], 0.05, 6.0, TARGET_ORACLE_AUC, 28
    )
    A = A_for_static(B)

    # alpha0 sets the base rate (AUCs are ~invariant to it near base 0.5).
    alpha0 = _bisect(lambda a0: _aucs(A, B, a0, q, eta, u)[0], -3.0, 3.0, TARGET_BASE_RATE, 34)

    # Back out world coefficients (D5.9). Fix kappa, sigma_h; split static's effect A into a
    # direct term (a_stat) and a health-seed term (a_h * s_mu).
    a_h = B / SIGMA_H
    a_stat = DIRECT_STATIC_FRACTION * A
    s_mu = (A - a_stat) / a_h  # so a_h * s_mu + a_stat == A
    sigma = SIGMA_H * float(
        np.sqrt(1.0 - (1.0 - KAPPA) ** 2)
    )  # sigma_h^2 = sigma^2/(1-(1-kappa)^2)

    return {
        "A": A,
        "B": B,
        "alpha0": alpha0,
        "alpha_h": -a_h,
        "alpha_stat": -a_stat,
        "s_mu": s_mu,
        "sigma": sigma,
        "sigma0": SIGMA_H,
    }


def evaluate_regime(params: dict, q_anchor: np.ndarray, rng: np.random.Generator, n: int) -> dict:
    """Recompute the achieved regime from the *world coefficients* (round-trip check)."""
    q, eta, u = _draw(q_anchor, n, rng)
    haz = params["hazard"]
    s_mu = params["quality_index"]["s_mu"]
    # h(t0) ~ mu(x) + sigma_h * eta at stationarity; mu(x) = s_mu * q.
    h = s_mu * q + params["latent"]["sigma0"] * eta
    p = K.hazard_prob(h, q, haz["alpha0"], haz["alpha_h"], haz["alpha_stat"])
    y = (u < p).astype(int)
    base = float(y.mean())
    static_auc = float(roc_auc_score(y, -q))
    oracle_auc = float(roc_auc_score(y, p))
    var_mu = float(np.var(s_mu * q))
    var_stoch = float(params["latent"]["sigma0"] ** 2)
    return {
        "base_rate": base,
        "static_auc": static_auc,
        "oracle_auc": oracle_auc,
        "recoverable_lift": oracle_auc - static_auc,
        "var_mu": var_mu,
        "var_stochastic_h": var_stoch,
        "var_split_mu_to_stochastic": f"1 : {var_stoch / var_mu:.2f}",
        "n": n,
    }


def build_params(coeffs: dict, num_stats: dict, q_raw_mean: float, q_raw_std: float) -> dict:
    """Assemble the full frozen params dict (no confirmatory seed in the clear -- D3)."""
    return {
        "schema_version": 1,
        "description": (
            "Frozen simulator world (PHASE0_LOCK_DECISIONS.md D5). Latent+hazard coefficients "
            "tuned by churn.simulator.tuning (scalar MC); event coefficients set a-priori. "
            "No confirmatory seed here -- only Stage-1 world constants."
        ),
        "latent": {
            "process": "ar1_ou_weekly",
            "weeks": 52,
            "kappa": KAPPA,
            "sigma": coeffs["sigma"],
            "sigma0": coeffs["sigma0"],
        },
        "quality_index": {
            "plan_score_map": PLAN_SCORE_MAP,
            "weights": QI_WEIGHTS,
            "zero_weight_features": ["region", "device_type"],
            "numeric_standardize": num_stats,
            "q_raw_standardize": {"mean": q_raw_mean, "std": q_raw_std},
            "s_mu": coeffs["s_mu"],
        },
        "hazard": {
            "form": "sigmoid(alpha0 + alpha_h*h + alpha_stat*q)",
            "alpha0": coeffs["alpha0"],
            "alpha_h": coeffs["alpha_h"],
            "alpha_stat": coeffs["alpha_stat"],
            "label_window_days": 90,
        },
        "events": EVENT_PARAMS,
        "temporal": {
            "feature_window_weeks": 52,
            "label_window_days": 90,
            "latent_step": "weekly",
            "row_key": ["customer_id", "t0"],
            "anchor_rows_per_customer": 1,
        },
        "population": {
            "size": 50_000,
            "anchor_path": "data/customer_data.csv",
            "anchor_n": 1600,
            "fit_marginals": [
                "region",
                "device_type",
                "subscription_plan",
                "account_age_days",
                "monthly_spend",
                "avg_order_value",
            ],
            "fit_joints": [
                ["subscription_plan", "monthly_spend"],
                ["subscription_plan", "account_age_days"],
                ["subscription_plan", "region"],
            ],
            "anchor_label_resimulated": True,
        },
        "floor_design": FLOOR_DESIGN,
        "controls": {
            "label_shuffle": {"method": "customer_level_permutation", "target": "y"},
            "null_stream": {
                "method": "zero_health_slopes",
                "zeroed_b": [
                    "login_rate",
                    "session_depth",
                    "order_rate",
                    "payment_fail",
                    "support_rate",
                    "sentiment",
                    "downgrade",
                ],
            },
        },
        "tuning": {
            "targets": {
                "base_rate": TARGET_BASE_RATE,
                "static_auc": TARGET_STATIC_AUC,
                "oracle_auc": TARGET_ORACLE_AUC,
            },
            "dev_seed": DEV_SEED,
            "search_n": SEARCH_N,
            "direct_static_fraction": DIRECT_STATIC_FRACTION,
            "note": "dev_seed is NOT the confirmatory seed (beacon-derived at Stage 2, D3).",
        },
    }


def main() -> None:
    static = load_anchor_static()
    num_stats = numeric_stats(static)
    q_raw = raw_quality(static, num_stats)
    q_raw_mean, q_raw_std = float(q_raw.mean()), float(q_raw.std(ddof=0))
    q_anchor = (q_raw - q_raw_mean) / q_raw_std  # unit-variance quality index over the anchor

    rng = np.random.default_rng(DEV_SEED)
    coeffs = solve_coefficients(q_anchor, rng)
    params = build_params(coeffs, num_stats, q_raw_mean, q_raw_std)

    report_rng = np.random.default_rng(DEV_SEED + 1)
    regime = evaluate_regime(params, q_anchor, report_rng, REPORT_N)

    SIM_PARAMS_PATH.write_text(
        json.dumps(registry.to_jsonable(params), indent=2) + "\n", newline="\n"
    )
    TUNING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_REPORT_PATH.write_text(
        json.dumps(registry.to_jsonable({"solved": coeffs, "achieved_regime": regime}), indent=2)
        + "\n",
        newline="\n",
    )

    print(f"Frozen world tuned. Achieved regime (report seed, N={REPORT_N}):")
    print(f"  base_rate        = {regime['base_rate']:.4f}  (target {TARGET_BASE_RATE:.2f})")
    print(f"  static_auc       = {regime['static_auc']:.4f}  (target {TARGET_STATIC_AUC:.2f})")
    print(f"  oracle_auc       = {regime['oracle_auc']:.4f}  (target {TARGET_ORACLE_AUC:.2f})")
    print(f"  recoverable_lift = {regime['recoverable_lift']:.4f}")
    print(f"  var split mu:stoch = {regime['var_split_mu_to_stochastic']}")
    print(f"Wrote {SIM_PARAMS_PATH.name}")
    print(f"Wrote reports/{TUNING_REPORT_PATH.name}")


if __name__ == "__main__":
    main()
