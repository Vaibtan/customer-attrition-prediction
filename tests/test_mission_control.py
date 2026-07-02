"""Mission-control data functions (pure; the Streamlit app just renders these)."""

from __future__ import annotations

import ast
from pathlib import Path

from services.dashboard import mission_control as MC


def test_performance_timeline_has_est_and_true():
    df = MC.performance_timeline(seed=1)
    assert {"true_auc", "cbpe_estimate", "retrained"} <= set(df.columns)
    assert len(df) > 5


def test_risk_tier_summary_partitions_the_cohort():
    summary = MC.risk_tier_summary(n=4000, seed=0)
    assert list(summary["risk_tier"]) == MC.TIERS
    assert summary["customers"].sum() == 4000
    assert (summary["customers"] > 0).all()


def test_promotion_history_records_wins_and_rejections():
    history = MC.promotion_history(seed=0)
    decisions = dict(zip(history["challenger"], history["decision"], strict=True))
    assert decisions["v3 (new features)"] == "PROMOTED"  # a genuine improvement wins
    assert decisions["v2 (retrain, noise)"] == "rejected"  # better-by-noise loses
    assert (history["decision"] == "rejected").sum() >= 2  # rejected challengers are shown


def test_streamlit_app_is_valid_python():
    # The UI is validated live by `streamlit run`; here just assert the app file parses.
    app = Path(__file__).resolve().parents[1] / "services" / "dashboard" / "app.py"
    ast.parse(app.read_text(encoding="utf-8"))
