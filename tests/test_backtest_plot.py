"""The centerpiece chart is generated as a real image artifact."""

from __future__ import annotations

from churn.backtest import plots, replay


def test_centerpiece_chart_is_written(tmp_path):
    df = replay.run_backtest(replay.BacktestConfig(seed=1))
    out = plots.plot_backtest(df, tmp_path / "chart.png")
    assert out.exists()
    assert out.stat().st_size > 2000  # a real rendered PNG, not an empty stub


def test_plot_cli_writes_default_figure(tmp_path):
    out = tmp_path / "centerpiece.png"
    plots.main(["--out", str(out), "--seed", "1"])
    assert out.exists()
