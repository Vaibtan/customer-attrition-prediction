"""Registry round-trip: a saved run reloads to an identical, usable model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from churn import registry
from churn.data import split_features_target
from churn.pipeline import build_pipeline


def test_save_and_load_run(tmp_path, sample):
    X, y, _ = split_features_target(sample)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {
        "model_name": "logistic_regression",
        "threshold": 0.4,
        "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.4},
    }

    run_dir = registry.save_run(model, meta, base_model=model, base_dir=tmp_path)
    assert (run_dir / registry.PIPELINE_FILE).exists()

    loaded, loaded_meta, base = registry.load_run(run_dir)
    # Reproducibility metadata is captured automatically.
    assert loaded_meta["model_name"] == "logistic_regression"
    assert "data_sha256" in loaded_meta
    assert set(loaded_meta["versions"]) == {"scikit_learn", "pandas", "numpy"}
    assert base is not None
    # The reloaded model scores identically.
    assert np.allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
    assert registry.latest_run_dir(tmp_path) == run_dir
