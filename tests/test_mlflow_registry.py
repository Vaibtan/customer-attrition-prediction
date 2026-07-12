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


def test_version_tags_ride_the_registration_and_resolve_alias_returns_them(tmp_path):
    """log_and_register stores VERSION tags (tier_cutpoints for serving) and resolve_alias
    hands back (version, tags, model) loaded by pinned version (ADR 0006)."""
    import json  # noqa: PLC0415

    uri = "sqlite:///" + str(tmp_path / "mlflow.db").replace("\\", "/")
    art = (tmp_path / "artifacts").as_uri()
    name = "churn_online_tags"
    cutpoints = {"t_mid": 0.3, "t_star": 0.6}

    v1 = MR.log_and_register(
        _model(1),
        name,
        {"roc_auc": 0.7},
        uri,
        artifact_location=art,
        tags={"tier_cutpoints": json.dumps(cutpoints)},
    )
    MR.set_alias(name, MR.CHAMPION, v1, uri)

    version, tags, model = MR.resolve_alias(name, MR.CHAMPION, uri)
    assert version == v1
    assert json.loads(tags["tier_cutpoints"]) == cutpoints
    assert hasattr(model, "predict_proba")


def test_champion_resolver_cutover_on_a_real_registry(tmp_path):
    """The full ADR 0006 loop on a real sqlite registry: promotion moves what serving reports."""
    import json  # noqa: PLC0415

    from churn.serving.champion import mlflow_alias_resolver  # noqa: PLC0415

    uri = "sqlite:///" + str(tmp_path / "mlflow.db").replace("\\", "/")
    art = (tmp_path / "artifacts").as_uri()
    name = "churn_online_cutover"
    tags = {"tier_cutpoints": json.dumps({"t_mid": 0.3, "t_star": 0.6})}

    v1 = MR.log_and_register(
        _model(1), name, {"roc_auc": 0.7}, uri, artifact_location=art, tags=tags
    )
    MR.set_alias(name, MR.CHAMPION, v1, uri)

    resolver = mlflow_alias_resolver(name, MR.CHAMPION, uri, ttl_seconds=0.0)
    assert resolver.run_id == f"{name}@v{v1}"
    assert resolver.health_status()["status"] == "ok"

    v2 = MR.log_and_register(
        _model(2), name, {"roc_auc": 0.8}, uri, artifact_location=art, tags=tags
    )
    MR.promote(name, v2, uri)
    assert resolver.run_id == f"{name}@v{v2}"  # the alias move IS the cutover
    assert resolver.health_status()["alias_version"] == v2


def test_resolver_without_tracking_uri_is_degraded_never_a_local_fallback():
    from churn.serving.champion import ModelUnavailable, mlflow_alias_resolver  # noqa: PLC0415

    resolver = mlflow_alias_resolver("any", MR.CHAMPION, None, ttl_seconds=0.0)
    import pytest as _pytest  # noqa: PLC0415

    with _pytest.raises(ModelUnavailable, match="MLFLOW_TRACKING_URI unset"):
        _ = resolver.run_id
    assert resolver.health_status()["status"] == "degraded"
