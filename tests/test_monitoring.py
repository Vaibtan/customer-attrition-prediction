"""Monitoring report smoke tests."""

from __future__ import annotations

import numpy as np

from churn import monitoring


def test_numeric_psi_is_zero_for_same_distribution(raw_full):
    psi = monitoring.numeric_psi(
        raw_full["days_since_last_login"], raw_full["days_since_last_login"]
    )
    assert np.isclose(psi, 0.0)


def test_data_quality_counts_known_invalid_rows(raw_full):
    quality = monitoring.data_quality_summary(raw_full)
    lookup = quality.set_index("column")

    assert lookup.loc["monthly_spend", "missing"] == 80
    assert lookup.loc["avg_order_value", "missing"] == 80
    assert lookup.loc["pages_per_session", "missing"] == 80
    assert lookup.loc["account_age_days", "invalid_range"] == 8
    assert lookup.loc["days_since_last_login", "invalid_range"] == 10
    assert "orders_zero_with_spend" not in quality["column"].tolist()


def test_build_drift_report_contains_core_sections(raw_full):
    report = monitoring.build_drift_report(raw_full, raw_full)
    assert "# Churn Monitoring Drift Report" in report
    assert "## Data Quality" in report
    assert "## Feature Drift" in report


def test_write_drift_report_without_registered_model(raw_full, tmp_path):
    current = tmp_path / "current.csv"
    out = tmp_path / "drift_report.md"
    raw_full.to_csv(current, index=False)

    written = monitoring.write_drift_report(current, current, out, run_dir=tmp_path / "missing")
    assert written == out
    assert out.exists()
    assert "Model run: not loaded" in out.read_text(encoding="utf-8")
