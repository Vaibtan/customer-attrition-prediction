"""Deployable online scoring model (static + event) + the online scorer (decision D8).

The online path's model is the instrument's ``static_event`` pipeline, fit on the synthetic
population's SIMULATED labels, with campaign-cost tier cutpoints. It is a synthetic-domain model:
the ``/score`` endpoint demonstrates the online serving path, it is not a real-world evidence claim
(ENHANCEMENT_PLAN.md Sec 1).

Serving reads the event-feature vector from Redis (populated by the streaming consumer) and the
static attributes from the request. Because online event features == offline PIT features (the
proven parity), the online score == the offline batch score for the same ``(customer, t0)`` -- the
train/serve-consistency guarantee, now at the *score* level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from churn import config, evaluate, registry
from churn.featurestore.cohort import make_cohort
from churn.instrument import model as M
from churn.scoring import score_to_tier

# The design matrix column order the pipeline expects (static attributes then PIT event features).
DESIGN_COLUMNS: list[str] = M.STATIC_FEATURES + M.EVENT_FEATURES
_NUMERIC_COLUMNS: list[str] = M.STATIC_NUMERIC + M.EVENT_FEATURES  # categoricals stay strings
MODEL_NAME = "online_static_event"


@dataclass(frozen=True)
class OnlineModel:
    """A fitted static+event pipeline plus its risk-tier cutpoints and source run id."""

    pipeline: Pipeline
    cutpoints: dict
    run_id: str | None = None


def train_online_model(dataset, pit_features: pd.DataFrame, seed: int = config.SEED) -> OnlineModel:
    """Fit the static+event pipeline on the synthetic labels; derive tier cutpoints from OOF EV."""
    design = M.assemble_design(dataset, pit_features, use_real_label=False)
    oof = M.group_oof_proba("static_event", design.X, design.y, design.groups, seed=seed)
    threshold = evaluate.CampaignThreshold.fit(design.y, oof, evaluate.Economics.default())
    pipeline = M.build_pipeline("static_event", seed=seed).fit(design.X[DESIGN_COLUMNS], design.y)
    return OnlineModel(pipeline=pipeline, cutpoints=threshold.cutpoints)


def assemble_row(static: Mapping[str, object], event: Mapping[str, object]) -> pd.DataFrame:
    """One-row design frame: static attributes + the online event-feature vector, in model order."""
    row = {c: static.get(c) for c in M.STATIC_FEATURES}
    row.update({c: event.get(c) for c in M.EVENT_FEATURES})
    frame = pd.DataFrame([row])[DESIGN_COLUMNS]
    # JSON null / None in a numeric column -> NaN float so the pipeline's imputer handles it
    # (a Python None would leave the column object-typed and break SimpleImputer).
    for col in _NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


@dataclass(frozen=True)
class OnlineScore:
    churn_probability: float
    risk_tier: str


class OnlineScorer:
    """Score one customer from static attributes + the Redis online event vector."""

    def __init__(self, model: OnlineModel) -> None:
        self.model = model

    @property
    def run_id(self) -> str | None:
        """Registry run id of the loaded model (None for a freshly-trained, unsaved model)."""
        return self.model.run_id

    def score(self, static: Mapping[str, object], event: Mapping[str, object]) -> OnlineScore:
        design_row = assemble_row(static, event)
        proba = float(self.model.pipeline.predict_proba(design_row)[:, 1][0])
        cut = self.model.cutpoints
        tier = score_to_tier(proba, cut["t_star"], cut["t_mid"])
        return OnlineScore(churn_probability=proba, risk_tier=str(tier))


def save_online_model(model: OnlineModel, base_dir: Path = config.MODELS_DIR) -> Path:
    """Persist the online model as a registry run (``pipeline.joblib`` + tier cutpoints)."""
    metadata = {"model_name": MODEL_NAME, "tier_cutpoints": model.cutpoints}
    return registry.save_run(model.pipeline, metadata, base_dir=base_dir)


def load_online_model(run_dir: Path | str | None = None) -> OnlineModel:
    """Load a persisted online model (pipeline + cutpoints) from a registry run directory."""
    loaded = registry.load_run(run_dir)
    return OnlineModel(pipeline=loaded.model, cutpoints=loaded.cutpoints, run_id=loaded.run_id)


def _build_full_dataset(seed: int, n_synthetic: int, t0: pd.Timestamp):
    from churn.featurestore import offline as OFF
    from churn.simulator import generate as G
    from churn.simulator import params as P

    dataset = G.build_population_dataset(P.load_params(), seed=seed, n_synthetic=n_synthetic)
    ids = dataset.customers["customer_id"].tolist()
    cohort = make_cohort(ids, t0)
    pit = OFF.compute_pit_features(dataset.events, cohort)
    return dataset, pit


def main(argv: list[str] | None = None) -> None:
    """CLI: train the online model on the full population and register it."""
    import argparse

    parser = argparse.ArgumentParser(description="Train + register the online static+event model.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-synthetic", type=int, default=0)
    parser.add_argument("--as-of", default="2025-01-01T00:00:00")
    args = parser.parse_args(argv)

    dataset, pit = _build_full_dataset(args.seed, args.n_synthetic, pd.Timestamp(args.as_of))
    model = train_online_model(dataset, pit, seed=args.seed)
    run_dir = save_online_model(model)
    print(f"Registered online model -> {run_dir}")


if __name__ == "__main__":
    main()
