"""Lightweight batch monitoring: data-quality, feature drift (PSI/KS), score drift."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, registry
from .data import load_data, validate_schema

REPORT_PATH = config.ROOT / "monitoring" / "drift_report.md"
MISSING_TOKEN = "__MISSING__"


def psi_from_counts(reference_counts, current_counts, eps: float = 1e-6) -> float:
    reference_counts = np.asarray(reference_counts, dtype="float64")
    current_counts = np.asarray(current_counts, dtype="float64")
    reference_pct = (reference_counts + eps) / (reference_counts.sum() + eps)
    current_pct = (current_counts + eps) / (current_counts.sum() + eps)
    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    return float(psi)


def aligned_level_counts(
    reference: pd.Series, current: pd.Series
) -> tuple[list, np.ndarray, np.ndarray]:
    """Align two categorical series onto one sorted level set (missing -> ``MISSING_TOKEN``).

    Returns ``(levels, ref_counts, cur_counts)`` -- integer counts reindexed onto the sorted union
    of levels, so PSI, the JS distance, and the chi-square contingency table all read the SAME
    aligned counts instead of each re-deriving them (and risking a mismatch).
    """
    ref = reference.astype("object").where(reference.notna(), MISSING_TOKEN)
    cur = current.astype("object").where(current.notna(), MISSING_TOKEN)
    levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts().reindex(levels, fill_value=0).to_numpy()
    cur_counts = cur.value_counts().reindex(levels, fill_value=0).to_numpy()
    return levels, ref_counts, cur_counts


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    _, ref_counts, cur_counts = aligned_level_counts(reference, current)
    return psi_from_counts(ref_counts, cur_counts)


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype="float64")
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return categorical_psi(pd.Series(ref), pd.Series(cur))

    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    return psi_from_counts(ref_counts, cur_counts)


def jensen_shannon_distance(reference: pd.Series, current: pd.Series) -> float:
    """JS distance (in [0, 1]) between two categorical distributions.

    The type-appropriate stat for categoricals: KS is undefined on unordered categories (the old
    report showed ``NaN``). For a per-feature significance test use ``drift.detectors`` (chi-
    square); this scalar keeps the legacy report's stat column meaningful for categoricals.
    """
    _, ref_counts, cur_counts = aligned_level_counts(reference, current)
    p = ref_counts.astype("float64")
    q = cur_counts.astype("float64")
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(jsd, 0.0)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Two-sample KS statistic without requiring scipy."""
    ref = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype="float64"))
    cur = np.sort(pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype="float64"))
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    values = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, values, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, values, side="right") / len(cur)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.RAW_FEATURE_COLUMNS:
        missing = int(df[col].isna().sum()) if col in df.columns else len(df)
        invalid = 0
        if col in config.VALID_RANGES and col in df.columns:
            lo, hi = config.VALID_RANGES[col]
            values = pd.to_numeric(df[col], errors="coerce")
            if lo is not None:
                invalid += int((values < lo).sum())
            if hi is not None:
                invalid += int((values > hi).sum())
        rows.append({"column": col, "missing": missing, "invalid_range": invalid})
    return pd.DataFrame(rows)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Per-feature drift with a TYPE-APPROPRIATE stat: KS for numerics, JSD for categoricals.

    ``stat`` no longer reports ``NaN`` for categoricals (KS is undefined on categories); ``test``
    records which statistic was used. For per-feature significance + a domain-classifier alarm, use
    ``drift.detectors.detect_drift``.
    """
    rows = []
    for col in config.RAW_FEATURE_COLUMNS:
        if col in config.BASE_NUMERIC:
            psi = numeric_psi(reference[col], current[col])
            stat, test = ks_statistic(reference[col], current[col]), "KS"
        else:
            psi = categorical_psi(reference[col], current[col])
            stat, test = jensen_shannon_distance(reference[col], current[col]), "JSD"
        rows.append({"feature": col, "psi": psi, "stat": stat, "test": test})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def model_scores(model, df: pd.DataFrame) -> np.ndarray:
    drop = [c for c in (config.ID_COL, config.TARGET) if c in df.columns]
    return model.predict_proba(df.drop(columns=drop))[:, 1]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in df[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model=None,
    run_id: str | None = None,
) -> str:
    validate_schema(reference_df, require_target=False)
    validate_schema(current_df, require_target=False)

    quality = data_quality_summary(current_df)
    drift = feature_drift(reference_df, current_df)

    lines = [
        "# Churn Monitoring Drift Report",
        "",
        f"- Reference rows: {len(reference_df):,}",
        f"- Current rows: {len(current_df):,}",
        f"- Model run: {run_id or 'not loaded'}",
        "",
        "## Data Quality",
        "",
        markdown_table(quality, ["column", "missing", "invalid_range"]),
        "",
        "## Feature Drift",
        "",
        markdown_table(drift, ["feature", "psi", "stat", "test"]),
    ]

    if model is not None:
        ref_scores = model_scores(model, reference_df)
        cur_scores = model_scores(model, current_df)
        score_psi = numeric_psi(pd.Series(ref_scores), pd.Series(cur_scores))
        lines.extend(
            [
                "",
                "## Prediction Drift",
                "",
                f"- Score PSI: {score_psi:.3f}",
                f"- Reference mean score: {np.mean(ref_scores):.3f}",
                f"- Current mean score: {np.mean(cur_scores):.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Alert Guide",
            "",
            "- PSI < 0.10: stable.",
            "- PSI 0.10-0.20: watch.",
            "- PSI > 0.20: investigate before trusting campaign decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_drift_report(
    reference_path=config.DATA_PATH,
    current_path=config.DATA_PATH,
    out_path=REPORT_PATH,
    run_dir=None,
) -> Path:
    reference_df = load_data(reference_path)
    current_df = load_data(current_path)

    model = None
    run_id = None
    try:
        loaded = registry.load_run(run_dir)
        model = loaded.model
        run_id = loaded.run_id
    except FileNotFoundError:
        pass

    report = build_drift_report(reference_df, current_df, model=model, run_id=run_id)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return out


def cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate a churn drift report.")
    parser.add_argument("--reference", default=str(config.DATA_PATH))
    parser.add_argument("--current", default=str(config.DATA_PATH))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)

    out = write_drift_report(args.reference, args.current, args.out, args.run_dir)
    print(f"Wrote drift report -> {out}")


if __name__ == "__main__":
    cli()
