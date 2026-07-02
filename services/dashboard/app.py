"""Streamlit ML mission-control.

Run with::

    streamlit run services/dashboard/app.py

A thin render layer over ``mission_control`` (which owns the tested data logic): risk tiers,
performance-under-drift (estimated vs true), and the promotion history including rejected
challengers. Ops metrics (latency/throughput/errors) live in Grafana, kept separate on purpose.
"""

from __future__ import annotations

import streamlit as st

from churn.backtest import plots
from services.dashboard import mission_control as MC

st.set_page_config(page_title="Churn ML Mission Control", layout="wide")
st.title("Churn ML Mission Control")
st.caption("Synthetic-domain systems demo -- not a real-world performance claim.")

timeline = MC.performance_timeline()

left, right = st.columns([2, 1])
with left:
    st.subheader("Estimated vs true performance under drift")
    st.pyplot(plots.build_figure(timeline))
    st.caption(
        "CBPE stays optimistic through concept drift (blind by design); the delayed-label monitor "
        "triggers a retrain that recovers true performance."
    )
with right:
    st.subheader("Risk tiers")
    tiers = MC.risk_tier_summary()
    st.bar_chart(tiers.set_index("risk_tier"), y="customers")

st.subheader("Promotion history (including rejected challengers)")
st.dataframe(MC.promotion_history(), use_container_width=True, hide_index=True)

st.subheader("Drift & retrain events")
st.dataframe(
    timeline[["step", "true_auc", "cbpe_estimate", "covariate_drift", "retrained"]],
    use_container_width=True,
    hide_index=True,
)
