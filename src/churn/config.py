"""Single source of truth for paths, schema, and business assumptions.

Every magic number that a reviewer might want to challenge lives here, not
scattered through the code. Paths are resolved from this file's location so the
project runs identically from any working directory (fixes the starter's
``../data`` bug).
"""

from __future__ import annotations

from pathlib import Path

# ── Paths (resolved from repo root, never from the cwd) ────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "customer_data.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

# ── Schema ─────────────────────────────────────────────────────────────────────
ID_COL = "customer_id"
TARGET = "churned"

CATEGORICAL_FEATURES = ["region", "device_type", "subscription_plan"]

BASE_NUMERIC = [
    "account_age_days",
    "monthly_spend",
    "num_orders_last_90d",
    "avg_order_value",
    "support_tickets_raised",
    "days_since_last_login",
    "pages_per_session",
]

# Engineered numeric columns emitted by FeatureEngineer (before the
# ColumnTransformer, so every name is present when it selects by name).
FEATURE_ENGINEER_OUTPUTS = ["recency_ratio", "is_dormant", "support_per_order"]

ENGINEERED_NUMERIC = FEATURE_ENGINEER_OUTPUTS

NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC

# All columns the pipeline expects to receive as raw input (id/target excluded).
RAW_FEATURE_COLUMNS = CATEGORICAL_FEATURES + BASE_NUMERIC

EXPECTED_COLUMNS = [ID_COL, *CATEGORICAL_FEATURES, *BASE_NUMERIC, TARGET]

# ── Domain validity ranges: values strictly outside (lo, hi) → NaN → imputed. ──
# None means unbounded. Derived from real-world meaning + the data audit
# (observed account age max ≈ 1496 days; login recency must lie in a 0–365 window;
# the 800/999 values are data-entry sentinels, negatives are impossible).
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    "account_age_days": (0, 3650),
    "monthly_spend": (0, None),
    "num_orders_last_90d": (0, None),
    "avg_order_value": (0, None),
    "support_tickets_raised": (0, None),
    "days_since_last_login": (0, 365),
    "pages_per_session": (0, None),
}

# ── Outlier winsorization (fit on the TRAIN fold only). ────────────────────────
# Only the genuinely spiky spend columns are capped; bounded columns and binary
# flags (e.g. is_dormant) are passed through untouched — winsorizing a rare flag
# would clip every positive to 0 (its p99 is 0). See data audit §2.4.
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
WINSORIZE_COLUMNS = ["monthly_spend", "avg_order_value"]

# ── Feature-engineering thresholds. ───────────────────────────────────────────
DORMANT_DAYS = 90  # days_since_last_login above which we call a customer dormant

# ── Business / cost model (ILLUSTRATIVE assumptions — supply real economics). ──
# Expected value of a campaign:
#   EV(t) = TP·uplift·value − (TP+FP)·cost
# A calibrated model is worth contacting customer i iff p_i exceeds the break-even
#   t_be = cost / (value · uplift).
# These are *placeholder* economics, NOT measured from the task data — the dataset
# carries no campaign costs. They stand in for "a modest outreach cost, a retention
# margin that dwarfs it, and a realistic single-digit-to-20% uplift". Every dollar
# figure downstream (t*, the EV headline, the tiers) is therefore CONDITIONAL on
# these inputs, so we never present it as a model performance result and instead
# report a sensitivity sweep over them (COST_SCENARIOS below, surfaced in the
# write-up). With the baseline values the break-even lands at 0.5 — that is a
# *consequence* of the numbers, not a target we reverse-engineered.
COST_PER_CONTACT = 20.0  # USD: retention offer/discount + outreach per contact
VALUE_PER_RETAINED = 200.0  # USD margin saved by retaining one would-be churner
CAMPAIGN_UPLIFT = 0.20  # fraction of contacted true churners actually retained

# Sensitivity scenarios: the operating point and EV are only as trustworthy as the
# economics, so we recompute t* and the hold-out EV across plausible regimes rather
# than headline a single dollar figure. (cost, value, uplift) per contact.
COST_SCENARIOS: dict[str, dict[str, float]] = {
    "cheap_contact": {"cost": 5.0, "value": 200.0, "uplift": 0.20},
    "baseline": {"cost": 20.0, "value": 200.0, "uplift": 0.20},
    "expensive_offer": {"cost": 50.0, "value": 200.0, "uplift": 0.20},
    "low_uplift": {"cost": 20.0, "value": 200.0, "uplift": 0.10},
    "high_clv": {"cost": 20.0, "value": 500.0, "uplift": 0.20},
}

# ── Model selection ────────────────────────────────────────────────────────────
# Bake-off models are ranked by CV ROC-AUC, but gaps within this tolerance are a
# statistical tie (here the top two sit ~0.002 apart — well inside the ~0.027
# fold-to-fold spread, i.e. roughly 1 SE of the mean). Ties are broken by the
# preference order below — simplest / most interpretable / best-calibrated first —
# NOT by chasing a third-decimal AUC that flips with the seed. A paired bootstrap
# of the hold-out AUC difference (whose CI straddles 0) backs the tie claim.
MODEL_SELECTION_TOLERANCE = 0.01
MODEL_PREFERENCE_ORDER = [
    "logistic_regression",  # linear, calibrated, interpretable, cheapest to serve
    "random_forest",
    "hist_gradient_boosting",
]
BOOTSTRAP_ROUNDS = 1000

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3
