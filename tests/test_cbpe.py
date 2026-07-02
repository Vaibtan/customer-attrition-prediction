"""CBPE: validated on a stable regime, provably blind to concept drift (centerpiece #4).

Two regimes with KNOWN answers: (1) no concept drift -> CBPE's estimate matches the realized AUC
within its band; (2) concept drift (feature->label relationship flipped) -> the model's scores are
unchanged so CBPE stays optimistic while the realized AUC collapses, and the realized value falls
OUTSIDE the CBPE band -- the limitation is demonstrated and flagged, not hidden.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from churn.drift import cbpe


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(0.0, 1.0, n)
    y = (x + rng.normal(0.0, 0.7, n) > 0).astype(int)  # x is genuinely predictive
    model = LogisticRegression().fit(x.reshape(-1, 1), y)
    return rng, x, y, model


def test_estimate_reads_confident_scores_as_high_and_uniform_as_mid():
    # Bimodal, confident scores -> a near-perfect ranker (~1). Uniform scores are NOT near-perfect:
    # perfectly-calibrated uniform probabilities self-consistently imply only ~0.83 AUC.
    confident = np.concatenate([np.full(2000, 0.99), np.full(2000, 0.01)])
    assert cbpe.estimate_auc(confident) > 0.95
    assert 0.80 < cbpe.estimate_auc(np.linspace(0.01, 0.99, 4000)) < 0.85


def test_cbpe_matches_realized_auc_without_concept_drift(fitted):
    rng, x, y, model = fitted
    proba = model.predict_proba(x.reshape(-1, 1))[:, 1]
    realized = roc_auc_score(y, proba)
    result = cbpe.estimate(proba, n_rounds=200, seed=1)
    assert abs(result.estimate - realized) < 0.03
    assert result.contains(realized)  # realized AUC sits inside the CBPE band
    assert cbpe.concept_drift_suspected(result, realized) is False


def test_cbpe_is_blind_under_concept_drift(fitted):
    rng, x, y, model = fitted
    proba = model.predict_proba(x.reshape(-1, 1))[:, 1]  # scores unchanged...
    # ...but the relationship flips: churn now follows -x. The model ranks by +x -> anti-correlated.
    y_drifted = (-x + rng.normal(0.0, 0.7, len(x)) > 0).astype(int)
    realized = roc_auc_score(y_drifted, proba)
    result = cbpe.estimate(proba, n_rounds=200, seed=2)

    assert result.estimate > 0.65  # CBPE stays optimistic (scores look just as confident)
    assert realized < 0.40  # the model is actually worse than random on the new regime
    assert result.estimate - realized > 0.2  # a large, telling gap
    assert cbpe.concept_drift_suspected(result, realized) is True  # flagged as blind
