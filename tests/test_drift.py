"""Type-aware drift detectors + drift simulation.

Pins: no false alarm on identical data; numeric covariate drift alarms (PSI + KS both fire) and the
domain classifier catches it; categorical drift yields a REAL statistic (never the old NaN KS);
prior shift moves the base rate; concept shift rewires feature->label while leaving the marginal
untouched (the case CBPE is blind to).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.drift import detectors as D
from churn.drift import simulate as S


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1]),
            "churned": rng.integers(0, 2, n),
        }
    )


def test_temporal_profiles_endpoints():
    assert S.sudden(0, 10) == 0.0
    assert S.sudden(9, 10) == 1.0
    assert S.gradual(0, 10) == 0.0
    assert S.gradual(9, 10) == 1.0
    assert S.recurring(0, 10) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= S.recurring(3, 10) <= 1.0


def test_no_drift_flags_nothing(base):
    report = D.detect_drift(base, base.copy(), numeric_features=["x"], categorical_features=["cat"])
    assert report.drifted_features == []
    assert report.covariate_shift is False


def test_numeric_covariate_drift_detected(base):
    current = S.covariate_shift(base, "x", magnitude=1.0, delta=1.5)  # +1.5 sigma shift
    report = D.detect_drift(base, current, numeric_features=["x"], categorical_features=["cat"])
    assert "x" in report.drifted_features
    assert report.covariate_shift is True
    xd = next(f for f in report.features if f.feature == "x")
    assert xd.kind == "numeric" and xd.p_value < 0.05 and xd.psi > D.PSI_THRESHOLD


def test_categorical_drift_has_real_stat_and_alarms(base):
    current = S.categorical_shift(base, "cat", magnitude=0.5, to_level="c", seed=1)
    report = D.detect_drift(base, current, numeric_features=["x"], categorical_features=["cat"])
    catd = next(f for f in report.features if f.feature == "cat")
    assert not np.isnan(catd.stat)  # the fix: categoricals get Cramer's V, never NaN
    assert catd.kind == "categorical"
    assert "cat" in report.drifted_features


def test_prior_shift_raises_base_rate(base):
    before = base["churned"].mean()
    shifted = S.prior_shift(base, "churned", magnitude=0.5, seed=2)
    assert shifted["churned"].mean() > before + 0.1


def test_concept_shift_preserves_marginal_but_rewires_label(base):
    shifted = S.concept_shift(base, "x", "churned", magnitude=1.0, seed=3)
    assert (shifted["x"].to_numpy() == base["x"].to_numpy()).all()  # marginal untouched
    corr = float(np.corrcoef(shifted["x"], shifted["churned"])[0, 1])
    assert corr > 0.3  # label now follows the feature (the relationship changed)
