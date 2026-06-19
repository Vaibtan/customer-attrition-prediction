"""Registry round-trip: a saved run reloads to an identical, usable model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from churn import config, registry
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def test_save_and_load_run(tmp_path, sample):
    X, y, ids = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {
        "model_name": "logistic_regression",
        "threshold": 0.4,
        "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.4},
    }

    run_dir = registry.save_run(model, meta, base_model=model, base_dir=tmp_path)
    assert (run_dir / registry.PIPELINE_FILE).exists()

    loaded = registry.load_run(run_dir)
    assert loaded.metadata["model_name"] == "logistic_regression"
    assert "data_sha256" in loaded.metadata
    assert set(loaded.metadata["versions"]) == {"scikit_learn", "pandas", "numpy"}
    assert loaded.base_linear is not None
    assert loaded.cutpoints == {"t_mid": 0.3, "t_star": 0.4}
    assert np.allclose(loaded.model.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
    assert registry.latest_run_dir(tmp_path) == run_dir


def test_loaded_model_scores_itself(tmp_path, sample):
    """LoadedModel.score owns the cutpoint plumbing callers used to repeat by hand."""
    X, y, ids = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {"model_name": "logistic_regression", "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.6}}
    run_dir = registry.save_run(model, meta, base_model=model, base_dir=tmp_path)

    scored = registry.load_run(run_dir).score(X.head(5))
    assert list(scored.columns) == [
        config.ID_COL,
        "churn_probability",
        "risk_tier",
        "top_reason_codes",
    ]
    assert len(scored) == 5
    assert set(scored["risk_tier"]) <= {"low", "medium", "high"}
