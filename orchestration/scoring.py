"""Phase-5 batch scoring: the weekly tick re-scores the frozen book with the pinned batch model.

``batch_scores`` is UNPARTITIONED and runs DOWNSTREAM of ``drift_gated_retrain`` inside the
existing ``weekly_retrain`` job -- the one legitimate recurrence in the frozen world (CONTEXT.md
"Weekly tick"). It is justified by the retrain decision upstream, never by pretending the frozen
data moved, so it rides that schedule rather than minting a scoring cron of its own.

Split source of truth (ADR 0006): batch scoring reads the PINNED file-registry model
(``registry.latest_run_dir()``), NEVER the MLflow ``@champion`` alias -- that alias governs the
online app; batch jobs pin artifacts and cut over by redeploy. And -- the honesty admission that is
part of the locked decision -- the upstream ``drift_gated_retrain`` branch is SCENARIO-DRIVEN and
does not move the registry champion: unless a human deliberately registers a newer run, the weekly
tick re-scores the same book with the same pinned model. Keying the artifact by ``run_id`` makes
that explicit -- the parquet only changes when the pinned model does.

(No ``from __future__ import annotations``: Dagster introspects real context annotations.)
"""

from pathlib import Path

from dagster import AssetExecutionContext, ConfigurableResource, MaterializeResult, asset

from orchestration.timeline import drift_gated_retrain


class BatchScoringConfig(ConfigurableResource):
    """Where the weekly tick reads truth from and writes scores to.

    Defaults are the repo's pinned registry + frozen book + ``reports/batch_scores``. Injectable so
    a host test can point at a tmp registry (CI ships no ``models/`` dir) and a tmp reports dir
    instead of mutating the repo tree.
    """

    models_dir: str | None = None  # None -> config.MODELS_DIR (the pinned batch registry)
    data_path: str | None = None  # None -> config.DATA_PATH (the frozen book)
    reports_dir: str | None = None  # None -> <ROOT>/reports/batch_scores


@asset(deps=[drift_gated_retrain])
def batch_scores(
    context: AssetExecutionContext, batch_scoring_config: BatchScoringConfig
) -> MaterializeResult:
    """Re-score the frozen book with the PINNED batch model; write a parquet + stamp provenance.

    Downstream of ``drift_gated_retrain`` but deliberately decoupled from it: it loads
    ``registry.latest_run_dir()`` (ADR 0006 pinned truth, NEVER ``@champion``), so a scenario-driven
    retrain that does not register a new run leaves these scores unchanged.
    """
    from churn import config, registry
    from churn.data import load_data

    models_dir = batch_scoring_config.models_dir or str(config.MODELS_DIR)
    data_path = batch_scoring_config.data_path or str(config.DATA_PATH)
    reports_dir = Path(
        batch_scoring_config.reports_dir or (config.ROOT / "reports" / "batch_scores")
    )

    run_dir = registry.latest_run_dir(models_dir)  # pinned batch truth, never the @champion alias
    loaded = registry.load_run(run_dir)
    book = load_data(data_path)
    scored = loaded.score(book)
    context.log.info(f"scored {len(scored)} customers with pinned run {run_dir.name}")

    proba = scored["churn_probability"]
    tiers = scored["risk_tier"].value_counts()
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Keyed by run_id: the same pinned model re-scores to the same file (the honesty note made
    # concrete -- a materialization only changes the artifact when the champion actually changed).
    artifact = reports_dir / f"{run_dir.name}.parquet"
    scored.to_parquet(artifact, index=False)

    return MaterializeResult(
        metadata={
            "run_id": loaded.run_id or run_dir.name,
            "data_sha256": registry.data_sha256(data_path),
            "rows": int(len(scored)),
            "artifact_path": str(artifact),
            "score_mean": float(proba.mean()),
            "score_std": float(proba.std()),
            "score_p50": float(proba.quantile(0.50)),
            "score_p90": float(proba.quantile(0.90)),
            "score_max": float(proba.max()),
            "n_low": int(tiers.get("low", 0)),
            "n_medium": int(tiers.get("medium", 0)),
            "n_high": int(tiers.get("high", 0)),
        }
    )
