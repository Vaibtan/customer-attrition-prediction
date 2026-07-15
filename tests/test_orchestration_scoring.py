"""Phase-5 batch scoring: the weekly tick re-scores the frozen book with the PINNED batch model.

Runs in-process via Dagster's ``materialize`` (no external infra). CI has no ``models/`` dir, so
the test pins its own small registry run and points ``BatchScoringConfig`` at it -- proving the
asset reads ``latest_run_dir()`` (batch-side pinned truth, ADR 0006), stamps provenance, and writes
a parquet artifact, and that the recurrence rides the existing weekly retrain (no scoring cron).
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("dagster")
pytest.importorskip("pyarrow")

from dagster import materialize  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from churn import registry  # noqa: E402
from churn.data import split_features_target  # noqa: E402
from churn.pipeline import build_pipeline  # noqa: E402
from orchestration import scoring as S  # noqa: E402
from orchestration import timeline as T  # noqa: E402


def _pin_a_batch_model(models_dir, raw_full):
    """Register a small pinned batch model (CI has no models/ dir). Fit on the full raw book so the
    OneHotEncoder has seen every category the asset will later score."""
    X, y, _ = split_features_target(raw_full)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    meta = {"model_name": "logistic_regression", "tier_cutpoints": {"t_mid": 0.3, "t_star": 0.6}}
    return registry.save_run(model, meta, base_model=model, base_dir=models_dir)


def test_batch_scores_stamps_provenance_and_writes_a_parquet(tmp_path, raw_full):
    run_dir = _pin_a_batch_model(tmp_path / "models", raw_full)
    reports = tmp_path / "reports"
    result = materialize(
        [S.batch_scores],
        resources={
            "batch_scoring_config": S.BatchScoringConfig(
                models_dir=str(tmp_path / "models"), reports_dir=str(reports)
            )
        },
    )
    assert result.success
    meta = result.asset_materializations_for_node("batch_scores")[0].metadata

    # Pinned batch truth (ADR 0006): the scored model is latest_run_dir(), never an MLflow alias.
    assert meta["run_id"].value == run_dir.name
    assert meta["rows"].value == len(raw_full)
    # Tiers partition every scored row -- the distribution stamp is complete, not a sample.
    assert meta["n_low"].value + meta["n_medium"].value + meta["n_high"].value == meta["rows"].value
    for key in ("data_sha256", "score_mean", "score_p50", "score_p90", "artifact_path"):
        assert key in meta

    artifact = reports / f"{run_dir.name}.parquet"
    assert artifact.exists()
    written = pd.read_parquet(artifact)
    assert len(written) == meta["rows"].value
    assert set(written["risk_tier"]) <= {"low", "medium", "high"}


def test_weekly_tick_runs_retrain_then_batch_scores(tmp_path, raw_full):
    """batch_scores is DOWNSTREAM of drift_gated_retrain: selecting both materialises the retrain
    branch then the score. The score is pinned to latest_run_dir() -- independent of the
    scenario-driven retrain, which does NOT move the registry champion (the locked honesty note)."""
    run_dir = _pin_a_batch_model(tmp_path / "models", raw_full)
    result = materialize(
        [T.drift_gated_retrain, S.batch_scores],
        resources={
            "retrain_scenario": T.RetrainScenario(inject_drift=False, n=800),
            "batch_scoring_config": S.BatchScoringConfig(
                models_dir=str(tmp_path / "models"), reports_dir=str(tmp_path / "reports")
            ),
        },
    )
    assert result.success
    meta = result.asset_materializations_for_node("batch_scores")[0].metadata
    assert meta["run_id"].value == run_dir.name  # scored the pinned model, not a retrain output


def test_no_scoring_cron_batch_rides_the_weekly_tick():
    from orchestration import definitions as D  # noqa: PLC0415

    # CONTEXT.md "Weekly tick": the recurrence belongs to the retrain schedule; adding batch
    # scoring must NOT mint a second cron.
    assert {s.name for s in D.defs.schedules} == {"weekly_retrain"}
    keys = {k.to_user_string() for k in D.defs.resolve_all_asset_keys()}
    assert "batch_scores" in keys
