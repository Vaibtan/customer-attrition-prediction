"""Smoke tests for figure generation (Agg backend, files actually written)."""

from __future__ import annotations

from churn import evaluate, plots


def test_eda_and_eval_plots_write_files(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(plots.config, "FIGURES_DIR", tmp_path)

    assert plots.eda_overview(sample).exists()
    assert plots.behavioural_by_churn(sample).exists()

    cm = evaluate.confusion_at([0, 1, 1, 0], [0.2, 0.9, 0.4, 0.6], 0.5)
    assert plots.confusion(cm, 0.5).exists()

    proba = [0.2, 0.9, 0.4, 0.6]
    y = [0, 1, 1, 0]
    assert plots.roc_curves({"logistic_regression": proba}, y).exists()
