"""Risk-tier stability — Codex finding #4.

A customer's tier must depend only on their own probability and the frozen
cutpoints, never on the composition of the scoring batch, and must be identical
whether scored alone (API) or in bulk (batch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn import config, registry
from churn.data import split_features_target
from churn.pipeline import build_pipeline
from churn.scoring import _cli, score_frame, score_to_tier


def test_score_to_tier_is_a_pure_function_of_cutpoints():
    assert score_to_tier(0.9, t_star=0.6, t_mid=0.3) == "high"
    assert score_to_tier(0.45, t_star=0.6, t_mid=0.3) == "medium"
    assert score_to_tier(0.1, t_star=0.6, t_mid=0.3) == "low"
    # Vectorised form agrees element-wise.
    arr = score_to_tier(np.array([0.9, 0.45, 0.1]), 0.6, 0.3)
    assert list(arr) == ["high", "medium", "low"]


def test_tier_invariant_to_batch_composition(sample):
    X, y, _ = split_features_target(
        sample.assign(**{config.ID_COL: range(len(sample)), config.TARGET: 0})
        if config.TARGET not in sample.columns
        else sample
    )
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)

    t_star, t_mid = 0.5, 0.3
    batch = score_frame(sample, model, t_star, t_mid)

    for i in (0, 7, 25, 100):
        single = score_frame(sample.iloc[[i]], model, t_star, t_mid)
        assert single["risk_tier"].iloc[0] == batch["risk_tier"].iloc[i]
        assert np.isclose(single["churn_probability"].iloc[0], batch["churn_probability"].iloc[i])


def test_probabilities_in_unit_interval(sample):
    X, y, _ = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    scored = score_frame(sample, model, 0.5, 0.3)
    p = scored["churn_probability"].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()


def test_scoring_cli_accepts_targetless_csv(sample, tmp_path):
    X, y, _ = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {
        "model_name": "logistic_regression",
        "threshold": 0.5,
        "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.5},
    }
    run_dir = registry.save_run(model, meta, base_model=model, base_dir=tmp_path / "models")

    raw_path = tmp_path / "targetless.csv"
    out_path = tmp_path / "scored.csv"
    sample.drop(columns=[config.TARGET]).head(10).to_csv(raw_path, index=False)

    _cli(["--in", str(raw_path), "--out", str(out_path), "--run-dir", str(run_dir)])

    scored = pd.read_csv(out_path)
    assert list(scored.columns) == [
        config.ID_COL,
        "churn_probability",
        "risk_tier",
        "top_reason_codes",
    ]
    assert len(scored) == 10
