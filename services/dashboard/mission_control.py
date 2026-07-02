"""Pure data functions for the ML mission-control (testable; the Streamlit app only renders these).

Separating the data from the UI keeps the interesting logic under test: the perf timeline (est vs
true), the risk-tier mix, and the promotion history INCLUDING rejected challengers -- the honest
record a reviewer wants (not just the winners).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.backtest import replay
from churn.lifecycle.promotion import evaluate_promotion
from churn.scoring import score_to_tier

TIERS = ["low", "medium", "high"]


def performance_timeline(seed: int = 1) -> pd.DataFrame:
    """Estimated (CBPE) vs true ROC-AUC over the drifting timeline, with drift/retrain markers."""
    return replay.run_backtest(replay.BacktestConfig(seed=seed))


def risk_tier_summary(
    n: int = 5000, t_mid: float = 0.30, t_star: float = 0.60, seed: int = 0
) -> pd.DataFrame:
    """Distribution of customers across risk tiers for a representative score cohort."""
    rng = np.random.default_rng(seed)
    proba = rng.beta(2.0, 5.0, n)  # right-skewed churn risk
    tiers = pd.Series(score_to_tier(proba, t_star, t_mid))
    counts = tiers.value_counts().reindex(TIERS, fill_value=0)
    return counts.rename_axis("risk_tier").reset_index(name="customers")


def promotion_history(seed: int = 0, n: int = 3000) -> pd.DataFrame:
    """Gate several challengers against the champion; record every decision (incl. rejections)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    segments = pd.DataFrame(
        {
            "subscription_plan": rng.choice(["Free", "Pro", "Premium"], n),
            "region": rng.choice(["North", "South"], n),
        }
    )

    def signal(strength: float, noise: float) -> np.ndarray:
        return np.clip(0.5 + strength * (y - 0.5) + rng.normal(0.0, noise, n), 1e-3, 1 - 1e-3)

    champion = signal(0.5, 0.20)
    challengers = {
        "v2 (retrain, noise)": np.clip(champion + rng.normal(0.0, 0.02, n), 1e-3, 1 - 1e-3),
        "v3 (new features)": signal(0.8, 0.12),
        "v4 (uncalibrated)": np.clip(signal(0.8, 0.10) * 0.3, 1e-3, 1 - 1e-3),
    }

    rows = []
    for name, challenger in challengers.items():
        decision = evaluate_promotion(y, champion, challenger, segments, n_rounds=200, seed=1)
        guards = {g.name: g.passed for g in decision.guardrails}
        rows.append(
            {
                "challenger": name,
                "roc_auc_lb": round(decision.delta_roc["diff_lo"], 4),
                "pr_auc_lb": round(decision.delta_pr["diff_lo"], 4),
                "calibration_ok": guards.get("calibration_brier", False),
                "segments_ok": guards.get("no_segment_degradation", False),
                "decision": "PROMOTED" if decision.promote else "rejected",
            }
        )
    return pd.DataFrame(rows)
