"""The centerpiece charts for the delayed-label backtest (``backtest.replay``).

Two stories, one harness:

- :func:`build_figure` (single run) -- true ROC-AUC vs the REVEALED (h-steps-late) AUC the
  monitor actually sees, vs the CBPE estimate with its band; drift region shaded, retrains
  marked. On a pure-concept run the CBPE/true gap in the trough IS the "blind to concept drift"
  limitation, shown rather than asserted.
- :func:`build_comparison_figure` (two runs, identical world) -- the label-free trigger recovers
  ~h steps before the lagged-label trigger: "label-free monitoring buys you the horizon".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to a file, never a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_TRUE = "#eb5757"
_REALIZED = "#f2994a"
_CBPE = "#2f80ed"
_RETRAIN = "#219653"
_DRIFT = "#f2c94c"
_FREE = "#219653"
_LAGGED = "#eb5757"


def _shade_drift(ax: plt.Axes, df: pd.DataFrame, label: str = "drift") -> None:
    drift = df.loc[df.get("angle", pd.Series(0, index=df.index)) > 0, "step"]
    if len(drift):
        ax.axvspan(drift.min() - 0.5, drift.max() + 0.5, color=_DRIFT, alpha=0.18, label=label)


def build_figure(
    df: pd.DataFrame, title: str = "Estimated vs true performance under drift"
) -> plt.Figure:
    """Build the single-run figure (for Streamlit ``st.pyplot`` or saving). Returns the Figure."""
    steps = df["step"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _shade_drift(ax, df, label="concept drift")

    ax.fill_between(steps, df["cbpe_lo"], df["cbpe_hi"], color=_CBPE, alpha=0.15)
    ax.plot(steps, df["cbpe_estimate"], color=_CBPE, marker="o", lw=2, label="CBPE (blind)")
    ax.plot(steps, df["true_auc"], color=_TRUE, marker="s", lw=2, label="true ROC-AUC")
    if "realized_auc" in df:
        ax.plot(
            steps,
            df["realized_auc"],
            color=_REALIZED,
            marker="^",
            lw=1.8,
            ls="--",
            label="revealed AUC (labels arrive late)",
        )

    retrain_label_used = False
    for s in df.loc[df["retrained"], "step"]:
        ax.axvline(
            s, color=_RETRAIN, ls="--", lw=1.6, label=None if retrain_label_used else "retrain"
        )
        retrain_label_used = True

    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("time step")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def build_comparison_figure(
    runs: dict[str, pd.DataFrame],
    title: str = "Label-free monitoring buys back the label horizon",
) -> plt.Figure:
    """Two trigger policies over the identical drifting world: recovery lead time made visible."""
    free, lagged = runs["label_free"], runs["lagged_label"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _shade_drift(ax, lagged)

    ax.plot(
        free["step"],
        free["true_auc"],
        color=_FREE,
        marker="o",
        lw=2,
        label="deployed AUC -- label-free trigger (drift detector)",
    )
    ax.plot(
        lagged["step"],
        lagged["true_auc"],
        color=_LAGGED,
        marker="s",
        lw=2,
        label="deployed AUC -- lagged-label trigger",
    )
    for df, color in ((free, _FREE), (lagged, _LAGGED)):
        for s in df.loc[df["retrained"], "step"]:
            ax.axvline(s, color=color, ls="--", lw=1.2, alpha=0.7)

    h = int(lagged["realized_auc"].isna().sum())  # rows before the first labels arrive
    ax.set_xlabel(f"time step (labels for step t arrive at t+{h})")
    ax.set_ylabel("ROC-AUC of the deployed champion")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_backtest(
    df: pd.DataFrame, out_path: str | Path, title: str = "Estimated vs true performance under drift"
) -> Path:
    """Render a single-run timeline to ``out_path`` (PNG). Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(df, title=title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_comparison(runs: dict[str, pd.DataFrame], out_path: str | Path) -> Path:
    """Render the trigger-comparison centerpiece to ``out_path`` (PNG). Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_comparison_figure(runs)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI: run the backtests and write both charts under reports/figures/."""
    import argparse

    from churn import config
    from churn.backtest import replay

    parser = argparse.ArgumentParser(description="Generate the backtest charts.")
    parser.add_argument("--out", default=str(config.FIGURES_DIR / "backtest_centerpiece.png"))
    parser.add_argument(
        "--blindspot-out", default=str(config.FIGURES_DIR / "backtest_blindspot.png")
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    # Centerpiece: combined drift, both triggers over the identical world.
    runs = replay.compare_triggers(replay.BacktestConfig(seed=args.seed, max_shift=1.5))
    out = plot_comparison(runs, args.out)
    print(f"Wrote centerpiece (trigger comparison) -> {out}")

    # Companion: pure concept drift, where label-free monitoring is structurally blind.
    blind = replay.run_backtest(replay.BacktestConfig(seed=args.seed))
    out2 = plot_backtest(
        blind,
        args.blindspot_out,
        title="Pure concept drift: label-free monitors are blind; labels arrive late",
    )
    print(f"Wrote blindspot chart -> {out2}")


if __name__ == "__main__":
    main()
