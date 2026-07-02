"""MLflow registry against a real sqlite-backed store (DB backend supports aliases; not a mock).

Pins the champion/challenger alias workflow: register two versions, name their roles by alias, load
the champion by alias (not by version), then promote -- an atomic alias move onto the challenger.
The HTTP-server topology is exercised separately in tests/test_mlflow_integration.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from churn.lifecycle import mlflow_registry as MR  # noqa: E402


def _model(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (200, 3))
    y = (x[:, 0] > 0).astype(int)
    return LogisticRegression().fit(x, y)


def test_register_alias_load_and_promote(tmp_path):
    uri = "sqlite:///" + str(tmp_path / "mlflow.db").replace("\\", "/")
    art = (tmp_path / "artifacts").as_uri()
    name = "churn_online"

    v1 = MR.log_and_register(_model(1), name, {"roc_auc": 0.70}, uri, artifact_location=art)
    MR.set_alias(name, MR.CHAMPION, v1, uri)
    v2 = MR.log_and_register(_model(2), name, {"roc_auc": 0.80}, uri, artifact_location=art)
    MR.set_alias(name, MR.CHALLENGER, v2, uri)

    assert MR.alias_version(name, MR.CHAMPION, uri) == v1
    assert MR.alias_version(name, MR.CHALLENGER, uri) == v2

    champion = MR.load_by_alias(name, MR.CHAMPION, uri)
    assert hasattr(champion, "predict_proba")

    MR.promote(name, v2, uri)
    assert MR.alias_version(name, MR.CHAMPION, uri) == v2
