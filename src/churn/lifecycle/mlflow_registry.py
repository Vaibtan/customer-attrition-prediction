"""MLflow tracking + model registry: log runs, register models, manage @champion/@challenger.

The lifecycle's system of record (ENHANCEMENT_PLAN.md Sec 5). A retrain logs a run and registers a
model VERSION; the registry's **aliases** name the roles: ``@champion`` is what serves and
``@challenger`` is the candidate. Promotion is an atomic alias move (champion := the challenger's
version), so serving code resolves ``models:/<name>@champion`` and never hard-codes a version.

Aliases require a database-backed store (sqlite or a tracking server) -- the file store does not
support them; that is why the tests use a sqlite/HTTP backend, never the default ``./mlruns`` files.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# One source for the alias names: serving (which must not hard-import mlflow) owns them.
from churn.serving.champion import CHALLENGER_ALIAS, CHAMPION_ALIAS

CHAMPION = CHAMPION_ALIAS
CHALLENGER = CHALLENGER_ALIAS


def _ensure_experiment(client: MlflowClient, name: str, artifact_location: str | None) -> str:
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(name, artifact_location=artifact_location)


def log_and_register(
    model,
    name: str,
    metrics: dict,
    tracking_uri: str,
    *,
    experiment: str = "churn",
    artifact_location: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a run (metrics + sklearn model), register a new model version, return that version.

    The version comes from the ``log_model`` return value itself -- never re-queried via
    ``search_model_versions`` + ``max()``, whose pagination/ordering could hand back a stale
    version under concurrency or long histories (REVIEW_ISSUES.md REV-13). ``tags`` are stored on
    the model VERSION (not the run), so an alias resolution can read them without a second hop --
    the serving path uses this for ``tier_cutpoints``.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(client, experiment, artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=name)
    version = str(info.registered_model_version)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, version, key, value)
    return version


def set_alias(name: str, alias: str, version: str, tracking_uri: str) -> None:
    MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(name, alias, version)


def alias_version(name: str, alias: str, tracking_uri: str) -> str:
    """Version currently holding ``alias``, always as ``str`` (MLflow 3.x hands back int)."""
    client = MlflowClient(tracking_uri=tracking_uri)
    return str(client.get_model_version_by_alias(name, alias).version)


def load_by_alias(name: str, alias: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def resolve_alias(name: str, alias: str, tracking_uri: str) -> tuple[str, dict[str, str], object]:
    """``(version, version_tags, model)`` for the alias holder, loaded by PINNED version.

    The alias is read once, then the artifact is loaded by that version -- not by alias again --
    so an alias move between the two reads cannot hand back tags from one version and the model
    from another.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    mv = client.get_model_version_by_alias(name, alias)
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    return str(mv.version), dict(mv.tags or {}), model


def promote(name: str, challenger_version: str, tracking_uri: str) -> None:
    """Atomically move ``@champion`` onto the challenger's version (the promotion effect)."""
    set_alias(name, CHAMPION, challenger_version, tracking_uri)
