"""The centerpiece chart: estimated (CBPE) vs true performance over a drifting timeline.

Draws the story ``backtest.replay`` produces -- true ROC-AUC (delayed labels) vs the CBPE estimate
with its band, the concept-drift region shaded, and retrain markers where the delayed-label monitor
recovers performance. The visible gap between CBPE and true in the trough IS the "blind to concept
drift" limitation, shown rather than asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to a file, never a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_TRUE = "#eb5757"
_CBPE = "#2f80ed"
_RETRAIN = "#219653"
_DRIFT = "#f2c94c"


def build_figure(
    df: pd.DataFrame, title: str = "Estimated vs true performance under drift"
) -> plt.Figure:
    """Build the centerpiece figure (for Streamlit ``st.pyplot`` or saving). Returns the Figure."""
    steps = df["step"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    drift = df.loc[df.get("angle", pd.Series(0, index=df.index)) > 0, "step"]
    if len(drift):
        ax.axvspan(
            drift.min() - 0.5, drift.max() + 0.5, color=_DRIFT, alpha=0.18, label="concept drift"
        )

    ax.fill_between(steps, df["cbpe_lo"], df["cbpe_hi"], color=_CBPE, alpha=0.15)
    ax.plot(steps, df["cbpe_estimate"], color=_CBPE, marker="o", lw=2, label="CBPE (blind)")
    ax.plot(steps, df["true_auc"], color=_TRUE, marker="s", lw=2, label="true ROC-AUC (labels)")

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


def plot_backtest(
    df: pd.DataFrame, out_path: str | Path, title: str = "Estimated vs true performance under drift"
) -> Path:
    """Render the backtest timeline to ``out_path`` (PNG). Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(df, title=title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI: run the backtest and write the centerpiece chart under reports/figures/."""
    import argparse

    from churn import config
    from churn.backtest import replay

    parser = argparse.ArgumentParser(description="Generate the backtest centerpiece chart.")
    parser.add_argument("--out", default=str(config.FIGURES_DIR / "backtest_centerpiece.png"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    df = replay.run_backtest(replay.BacktestConfig(seed=args.seed))
    out = plot_backtest(df, args.out)
    print(f"Wrote centerpiece chart -> {out}")


if __name__ == "__main__":
    main()
