"""The centerpiece chart is generated as a real image artifact."""

from __future__ import annotations

from churn.backtest import plots, replay


def test_single_run_chart_is_written(tmp_path):
    df = replay.run_backtest(replay.BacktestConfig(seed=1))
    out = plots.plot_backtest(df, tmp_path / "chart.png")
    assert out.exists()
    assert out.stat().st_size > 2000  # a real rendered PNG, not an empty stub


def test_comparison_chart_is_written(tmp_path):
    runs = replay.compare_triggers(replay.BacktestConfig(seed=1, max_shift=1.5))
    out = plots.plot_comparison(runs, tmp_path / "comparison.png")
    assert out.exists()
    assert out.stat().st_size > 2000


def test_plot_cli_writes_both_figures(tmp_path):
    out = tmp_path / "centerpiece.png"
    blind = tmp_path / "blindspot.png"
    plots.main(["--out", str(out), "--blindspot-out", str(blind), "--seed", "1"])
    assert out.exists()
    assert blind.exists()
