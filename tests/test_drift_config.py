"""ISS-11: monitoring policy has one source -- config PSI bands + declared-schema partition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config, monitoring
from churn.drift import detectors


def test_psi_threshold_and_alert_guide_share_the_config_band():
    assert detectors.PSI_THRESHOLD == config.PSI_INVESTIGATE
    report = monitoring.build_drift_report(
        pd.read_csv(config.DATA_PATH), pd.read_csv(config.DATA_PATH)
    )
    assert f"PSI > {config.PSI_INVESTIGATE:.2f}: investigate" in report
    assert f"PSI < {config.PSI_WATCH:.2f}: stable" in report


def test_detect_drift_partitions_known_columns_by_declared_schema_not_dtype():
    """A numerically-coded categorical must be treated as categorical (declared type wins)."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            # subscription_plan numerically coded: dtype sniffing would call it numeric.
            "subscription_plan": rng.integers(0, 3, size=n),
            "monthly_spend": rng.gamma(2.0, 30.0, size=n),
            # Unknown column: falls back to dtype sniffing (backtest synthetic case).
            "f0": rng.normal(size=n),
        }
    )
    report = detectors.detect_drift(frame, frame.copy())
    kinds = {f.feature: f.kind for f in report.features}
    assert kinds["subscription_plan"] == "categorical"  # declared, despite int dtype
    assert kinds["monthly_spend"] == "numeric"
    assert kinds["f0"] == "numeric"  # sniffed fallback
