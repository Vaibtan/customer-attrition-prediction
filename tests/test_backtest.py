"""Backtest replay harness -- the delayed-label narrative, pinned (REV-02).

Labels arrive ``h`` steps late: the monitor sees ``realized_auc`` (the truth about ``t - h``),
never ``true_auc`` at ``t``. Under pure concept drift the label-free signals are structurally
blind and the lagged monitor recovers late; under combined drift the label-free trigger recovers
~h steps earlier over the identical world.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.backtest import replay as R

H = R.BacktestConfig().label_horizon_steps


def test_realized_auc_is_the_truth_about_t_minus_h():
    df = R.run_backtest(R.BacktestConfig(seed=1))
    assert df["realized_auc"].isna().sum() == H  # nothing is labeled before step h
    got = df["realized_auc"].to_numpy()[H:]
    expected = df["true_auc"].to_numpy()[:-H]
    np.testing.assert_allclose(got, expected)


def test_concept_drift_story_with_an_honest_lag():
    cfg = R.BacktestConfig(seed=1)
    df = R.run_backtest(cfg)
    for col in (
        "step",
        "true_auc",
        "realized_auc",
        "cbpe_estimate",
        "cbpe_lo",
        "cbpe_hi",
        "covariate_drift",
        "retrained",
    ):
        assert col in df.columns

    # Champion is strong before drift; concept drift decays true performance past the threshold.
    assert df.loc[df["step"] < cfg.drift_start, "true_auc"].min() > 0.85
    assert df["true_auc"].min() < cfg.retrain_threshold

    # The retrain can only fire once the DECAYED window's labels have arrived: strictly after
    # the true decay crossed the threshold, by at least the horizon.
    assert df["retrained"].any()
    first_decay = int(df.index[df["true_auc"] < cfg.retrain_threshold][0])
    first_retrain = int(df.index[df["retrained"]][0])
    assert first_retrain >= first_decay + H

    # ... and performance recovers after retraining on the latest LABELED window.
    assert df.loc[first_retrain + 1 :, "true_auc"].max() > 0.85

    # CBPE is BLIND to concept drift: in the trough it stays well above the realized truth.
    trough = df[df["true_auc"] < cfg.retrain_threshold]
    assert (trough["cbpe_estimate"] - trough["true_auc"] > 0.15).any()

    # Pure concept drift is invisible to covariate detectors (marginals never move).
    assert not df["covariate_drift"].any()


def test_label_free_trigger_recovers_earlier_on_the_identical_combined_world():
    runs = R.compare_triggers(R.BacktestConfig(seed=1, max_shift=1.5))
    free, lagged = runs["label_free"], runs["lagged_label"]

    # Identical world: the ground-truth decay is the same series until policies diverge by
    # retraining -- the PRE-retrain prefix must match exactly.
    first_action = min(
        int(free.index[free["retrained"]][0]), int(lagged.index[lagged["retrained"]][0])
    )
    np.testing.assert_allclose(
        free.loc[: first_action - 1, "true_auc"], lagged.loc[: first_action - 1, "true_auc"]
    )

    # Combined drift moves the marginals -> the detector fires (unlike the pure-concept run).
    assert free["covariate_drift"].any()

    # The label-free trigger acts before the lagged-label trigger CAN act.
    free_first = int(free.index[free["retrained"]][0])
    lagged_first = int(lagged.index[lagged["retrained"]][0])
    assert free_first < lagged_first

    # And earlier action buys a healthier timeline: no worse at every step, strictly better
    # somewhere in the recovery window.
    diff = free["true_auc"] - lagged["true_auc"]
    assert diff.loc[free_first:lagged_first].max() > 0.02


def test_unknown_trigger_is_rejected():
    with pytest.raises(ValueError, match="unknown trigger"):
        R.run_backtest(R.BacktestConfig(trigger="psychic"))
