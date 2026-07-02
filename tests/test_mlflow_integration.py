"""MLflow champion/challenger against a LIVE MLflow tracking + registry server (HTTP topology).

The deployment shape: a client with only ``MLFLOW_TRACKING_URI=http://mlflow:5000`` logs models,
registers versions, and resolves ``models:/<name>@champion`` -- artifacts proxied by the server. A
fresh registered-model name per run keeps the shared server's state isolated.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("mlflow")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from churn.lifecycle import mlflow_registry as MR  # noqa: E402

pytestmark = pytest.mark.integration


def _model(seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (200, 3))
    y = (x[:, 0] > 0).astype(int)
    return LogisticRegression().fit(x, y)


def test_champion_challenger_lifecycle_on_live_server(mlflow_uri):
    name = f"churn_it_{uuid.uuid4().hex[:8]}"

    v1 = MR.log_and_register(_model(1), name, {"roc_auc": 0.70}, mlflow_uri)
    MR.set_alias(name, MR.CHAMPION, v1, mlflow_uri)
    v2 = MR.log_and_register(_model(2), name, {"roc_auc": 0.80}, mlflow_uri)
    MR.set_alias(name, MR.CHALLENGER, v2, mlflow_uri)

    assert MR.alias_version(name, MR.CHAMPION, mlflow_uri) == v1
    assert MR.alias_version(name, MR.CHALLENGER, mlflow_uri) == v2

    # Serving resolves the champion by ALIAS, never by version.
    champion = MR.load_by_alias(name, MR.CHAMPION, mlflow_uri)
    assert hasattr(champion, "predict_proba")

    # Promotion is an atomic alias move onto the challenger's version.
    MR.promote(name, v2, mlflow_uri)
    assert MR.alias_version(name, MR.CHAMPION, mlflow_uri) == v2
