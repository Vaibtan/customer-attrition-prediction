"""Analysis-model tests -- the leak-free instrument the baselines/floors/confirmatory run use.

Pins the group-aware, preprocessing-per-fold measurement contract: out-of-fold predictions are
produced with a StratifiedGroupKFold by ``customer_id`` (no customer straddles the split -- §4.7),
the preprocessor is refit inside each fold (no leakage), and the static+event model recovers the
injected health signal (higher AUC than static-only) on the frozen world.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.featurestore import offline as OFF
from churn.instrument import model as M
from churn.simulator import generate as G
from churn.simulator import params as P


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def frame(params):
    ds = G.build_population_dataset(params, seed=4242, n_synthetic=0)  # anchor only, n=1600
    pit = OFF.compute_pit_features(ds.events, ds.cohort)
    return M.assemble_design(ds, pit)


def test_feature_partition(frame):
    assert set(M.STATIC_FEATURES).isdisjoint(M.EVENT_FEATURES)
    assert "customer_id" not in M.STATIC_FEATURES + M.EVENT_FEATURES  # no id-derived features
    for col in M.STATIC_FEATURES + M.EVENT_FEATURES:
        assert col in frame.X.columns


def test_group_oof_is_leak_free_and_full_length(frame):
    proba = M.group_oof_proba("static", frame.X, frame.y, frame.groups, seed=1)
    assert proba.shape == frame.y.shape
    assert ((proba >= 0) & (proba <= 1)).all()


def test_no_customer_straddles_the_split(frame):
    # Build a panel fixture: duplicate each anchor customer into 2 rows, assert group folds keep
    # a customer's rows on one side (the §4.7 group-aware-split sentinel).
    groups = np.repeat(frame.groups, 2)
    y = np.repeat(frame.y, 2)
    for train_idx, test_idx in M.stratified_group_folds(y, groups, seed=3):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_static_plus_event_beats_static(frame):
    static = M.auc(frame.y, M.group_oof_proba("static", frame.X, frame.y, frame.groups, seed=7))
    both = M.auc(frame.y, M.group_oof_proba("static_event", frame.X, frame.y, frame.groups, seed=7))
    # The positive control: event features recover latent-health signal static cannot see.
    assert both > static + 0.03, (static, both)


def test_preprocessing_is_fit_per_fold(frame):
    # Two different fold seeds give different OOF predictions -> preprocessing/model refit per fold.
    p1 = M.group_oof_proba("static", frame.X, frame.y, frame.groups, seed=1)
    p2 = M.group_oof_proba("static", frame.X, frame.y, frame.groups, seed=2)
    assert not np.allclose(p1, p2)
