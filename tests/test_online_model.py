"""Online scoring model (static + event) -- pure host tests (no infra).

The deployable online model is the instrument's static+event pipeline fit on the synthetic
population. These tests pin: it trains + scores in range; the online scorer's probability equals a
direct pipeline apply on the assembled design row (so ``assemble_row`` reconstructs the design); and
it survives a registry save/load round-trip.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.featurestore import offline as OFF
from churn.instrument import model as M
from churn.serving import online_model as OM
from churn.simulator import generate as G
from churn.simulator import params as P

T0 = pd.Timestamp("2025-01-01T00:00:00")


@pytest.fixture(scope="module")
def trained():
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = ds.customers["customer_id"].head(500).tolist()
    cohort = pd.DataFrame({"customer_id": ids, "t0": [T0] * len(ids)})
    pit = OFF.compute_pit_features(ds.events, cohort)
    model = OM.train_online_model(ds, pit, seed=42)
    return ds, pit.set_index("customer_id"), model


def _static_event(ds, pit_indexed, cid):
    row = ds.customers.set_index("customer_id").loc[cid]
    static = {k: row[k] for k in M.STATIC_FEATURES}
    event = {c: float(pit_indexed.loc[cid, c]) for c in OFF.FEATURE_COLUMNS}
    return static, event


def test_trains_and_scores_in_range(trained):
    ds, pit, model = trained
    scorer = OM.OnlineScorer(model)
    static, event = _static_event(ds, pit, pit.index[0])
    result = scorer.score(static, event)
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.risk_tier in {"low", "medium", "high"}


def test_scorer_matches_direct_pipeline_apply(trained):
    ds, pit, model = trained
    scorer = OM.OnlineScorer(model)
    static, event = _static_event(ds, pit, pit.index[3])
    design_row = OM.assemble_row(static, event)
    assert list(design_row.columns) == OM.DESIGN_COLUMNS
    direct = float(model.pipeline.predict_proba(design_row)[:, 1][0])
    assert scorer.score(static, event).churn_probability == pytest.approx(direct, abs=1e-12)


def test_save_load_round_trip(trained, tmp_path):
    ds, pit, model = trained
    run_dir = OM.save_online_model(model, base_dir=tmp_path)
    loaded = OM.load_online_model(run_dir)
    static, event = _static_event(ds, pit, pit.index[5])
    a = OM.OnlineScorer(model).score(static, event)
    b = OM.OnlineScorer(loaded).score(static, event)
    assert a.churn_probability == pytest.approx(b.churn_probability, abs=1e-12)
    assert a.risk_tier == b.risk_tier
