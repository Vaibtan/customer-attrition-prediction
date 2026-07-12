"""Single source of truth for paths, schema, and business assumptions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "customer_data.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

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

FEATURE_ENGINEER_OUTPUTS = ["recency_ratio", "is_dormant", "support_per_order"]
ENGINEERED_NUMERIC = FEATURE_ENGINEER_OUTPUTS
NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC
RAW_FEATURE_COLUMNS = CATEGORICAL_FEATURES + BASE_NUMERIC
EXPECTED_COLUMNS = [ID_COL, *CATEGORICAL_FEATURES, *BASE_NUMERIC, TARGET]

# Values strictly outside (lo, hi) are mapped to NaN and imputed inside the pipeline.
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    "account_age_days": (0, 3650),
    "monthly_spend": (0, None),
    "num_orders_last_90d": (0, None),
    "avg_order_value": (0, None),
    "support_tickets_raised": (0, None),
    "days_since_last_login": (0, 365),
    "pages_per_session": (0, None),
}

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
WINSORIZE_COLUMNS = ["monthly_spend", "avg_order_value"]

DORMANT_DAYS = 90

# Illustrative campaign economics (not measured from the data); see WRITEUP Part 5.
COST_PER_CONTACT = 20.0
VALUE_PER_RETAINED = 200.0
CAMPAIGN_UPLIFT = 0.20

COST_SCENARIOS: dict[str, dict[str, float]] = {
    "cheap_contact": {"cost": 5.0, "value": 200.0, "uplift": 0.20},
    "baseline": {"cost": 20.0, "value": 200.0, "uplift": 0.20},
    "expensive_offer": {"cost": 50.0, "value": 200.0, "uplift": 0.20},
    "low_uplift": {"cost": 20.0, "value": 200.0, "uplift": 0.10},
    "high_clv": {"cost": 20.0, "value": 500.0, "uplift": 0.20},
}

# Models within this CV-AUC tolerance of the best are a tie, broken by preference order.
MODEL_SELECTION_TOLERANCE = 0.01
MODEL_PREFERENCE_ORDER = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
]
BOOTSTRAP_ROUNDS = 1000

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3

# --- Monitoring policy (ISS-11: one source for the PSI bands) ----------------------------------
# The classic PSI ladder; referenced by BOTH drift.detectors (alarm threshold) and the
# monitoring-report Alert Guide so the prose can never disagree with the alarm again.
PSI_WATCH = 0.10  # PSI in [WATCH, INVESTIGATE): distribution moving, keep an eye on it
PSI_INVESTIGATE = 0.20  # PSI > this: investigate before trusting campaign decisions

# --- Promotion policy (ENHANCEMENT_PLAN.md Sec 4.8) -------------------------------------------
# A challenger is promoted only if BOTH paired lower bounds (ROC-AUC and PR-AUC) exceed the MDE
# (not merely > 0), and every guardrail passes; the incumbent wins ties. EV is sensitivity only.
PROMOTION_MDE = 0.005  # minimum detectable effect (AUC/PR-AUC points) the LB must clear
PROMOTION_ALPHA = 0.05  # 95% paired-bootstrap CI
PROMOTION_BRIER_TOLERANCE = 0.02  # challenger Brier may not exceed champion Brier by more than this
PROMOTION_SEGMENT_TOLERANCE = 0.02  # per-segment ROC-AUC may not drop by more than this
PROMOTION_SEGMENTS = ["subscription_plan", "region"]  # declared no-degradation segments
PROMOTION_MIN_SEGMENT_ROWS = 50  # skip AUC on segments too small to estimate
