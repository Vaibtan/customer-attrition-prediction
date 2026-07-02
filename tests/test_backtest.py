"""Backtest replay harness -- the centerpiece narrative, pinned.

Concept drift decays true performance while CBPE stays optimistic (blind) and covariate detectors
stay silent; a delayed-label monitor triggers a retrain that recovers performance. These are the
exact series the centerpiece chart draws.
"""

from __future__ import annotations

from churn.backtest import replay as R


def test_backtest_timeline_tells_the_drift_and_recovery_story():
    df = R.run_backtest(R.BacktestConfig(seed=1))
    expected = ("step", "true_auc", "cbpe_estimate", "cbpe_lo", "cbpe_hi", "covariate_drift")
    for col in (*expected, "retrained"):
        assert col in df.columns

    # Champion is strong before drift.
    assert df.loc[df["step"] < R.BacktestConfig().drift_start, "true_auc"].min() > 0.85

    # Concept drift decays true performance past the retrain threshold.
    assert df["true_auc"].min() < 0.72

    # The delayed-label monitor fires a retrain, and performance recovers afterwards.
    assert df["retrained"].any()
    first_retrain = int(df.index[df["retrained"]][0])
    assert df.loc[first_retrain + 1 :, "true_auc"].max() > 0.85

    # CBPE is BLIND to concept drift: in the trough it stays well above the realized AUC.
    trough = df[df["true_auc"] < 0.72]
    assert (trough["cbpe_estimate"] - trough["true_auc"] > 0.15).any()

    # Concept drift is invisible to covariate detectors (marginals never move).
    assert not df["covariate_drift"].any()
