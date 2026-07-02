"""Cohort-diagnostics tests (ENHANCEMENT_PLAN.md §4.5).

Before any blended claim, report anchor vs synthetic side by side: base rate, categorical marginals,
numeric summaries + missingness, and static-only AUC. These *scope* a result to a domain; they are
not a validity guarantee. The test checks the diagnostic is structurally complete and that the
synthetic cohort is production-shaped (base rate + plan marginal track the anchor).
"""

from __future__ import annotations

import pytest

from churn.featurestore import offline as OFF
from churn.instrument import diagnostics as D
from churn.simulator import generate as G
from churn.simulator import params as P


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def report(params):
    ds = G.build_population_dataset(params, seed=2026, n_synthetic=6000)
    pit = OFF.compute_pit_features(ds.events, ds.cohort)
    return D.cohort_diagnostics(ds, pit, seed=1, synthetic_cap=4000)


def test_report_is_structurally_complete(report):
    for cohort in ("anchor", "synthetic"):
        c = report[cohort]
        assert c["n"] > 0
        assert 0.0 <= c["base_rate"] <= 1.0
        assert set(c["categorical_marginals"]) == {"region", "device_type", "subscription_plan"}
        assert "monthly_spend" in c["numeric_summary"]
        assert "static_auc" in c


def test_synthetic_is_production_shaped(report):
    a, s = report["anchor"], report["synthetic"]
    # Both cohorts are drawn from the same frozen world -> comparable base rates.
    assert abs(a["base_rate"] - s["base_rate"]) < 0.06
    # The plan marginal is fitted to the anchor.
    a_plan = a["categorical_marginals"]["subscription_plan"]
    s_plan = s["categorical_marginals"]["subscription_plan"]
    for plan, frac in a_plan.items():
        assert abs(frac - s_plan.get(plan, 0.0)) < 0.05


def test_missingness_is_reported(report):
    # The anchor has real missing spend/aov (~5%); synthetic is complete (D6.4) -- disclosed here.
    assert report["anchor"]["numeric_summary"]["monthly_spend"]["missing_frac"] > 0.0
    assert report["synthetic"]["numeric_summary"]["monthly_spend"]["missing_frac"] == 0.0
