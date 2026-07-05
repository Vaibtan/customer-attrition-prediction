"""Promotion gate (ENHANCEMENT_PLAN.md Sec 4.8) -- statistical teeth, no false precision.

A challenger replaces the champion only if it clears a **two-sided** bar:

1. **Primary gate** -- the paired-bootstrap lower bounds of BOTH ΔROC-AUC and ΔPR-AUC exceed a
   minimum detectable effect (``PROMOTION_MDE``), not merely 0. Same resamples for both metrics.
2. **Guardrails** -- no calibration regression (Brier within tolerance) and no per-segment
   degradation on the declared segments (``subscription_plan``, ``region``).

The **incumbent wins ties**: anything short of clearing the MDE margin AND passing every guardrail
leaves the champion in place. Expected value across cost scenarios is reported elsewhere as
sensitivity and is never a gate. The centerpiece guarantee (tested): a *better-by-noise* challenger
is not promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from churn import config, evaluate


@dataclass(frozen=True)
class GuardrailResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    """The gate's verdict, with every input to the decision surfaced for the audit trail."""

    promote: bool
    primary_passed: bool
    delta_roc: dict
    delta_pr: dict
    guardrails: list[GuardrailResult]
    reasons: list[str]


def _brier_guardrail(y, champ, chall, tol: float) -> GuardrailResult:
    b_champ = float(brier_score_loss(y, champ))
    b_chall = float(brier_score_loss(y, chall))
    passed = b_chall <= b_champ + tol
    return GuardrailResult(
        "calibration_brier",
        passed,
        f"champion Brier {b_champ:.4f} -> challenger {b_chall:.4f} (tol {tol})",
    )


def _segment_guardrail(
    y, champ, chall, segment_frame, columns, tol: float, min_rows: int
) -> GuardrailResult:
    degradations: list[tuple[str, object, float]] = []
    if segment_frame is not None:
        for col in columns:
            if col not in segment_frame.columns:
                continue
            values = segment_frame[col].to_numpy()
            for level in pd.unique(segment_frame[col].dropna()):
                mask = values == level
                if mask.sum() < min_rows or np.unique(y[mask]).size < 2:
                    continue
                a_champ = roc_auc_score(y[mask], champ[mask])
                a_chall = roc_auc_score(y[mask], chall[mask])
                if a_chall < a_champ - tol:
                    degradations.append((col, level, float(a_champ - a_chall)))
    if not degradations:
        return GuardrailResult("no_segment_degradation", True, "no segment degraded past tolerance")
    col, level, drop = max(degradations, key=lambda d: d[2])
    detail = f"{len(degradations)} segment(s) degraded; worst {col}={level} ROC-AUC drop {drop:.3f}"
    return GuardrailResult("no_segment_degradation", False, detail)


def evaluate_promotion(
    y,
    champion_proba,
    challenger_proba,
    segment_frame: pd.DataFrame | None = None,
    *,
    mde: float = config.PROMOTION_MDE,
    alpha: float = config.PROMOTION_ALPHA,
    brier_tol: float = config.PROMOTION_BRIER_TOLERANCE,
    segment_tol: float = config.PROMOTION_SEGMENT_TOLERANCE,
    segment_columns: list[str] | None = None,
    min_segment_rows: int = config.PROMOTION_MIN_SEGMENT_ROWS,
    n_rounds: int = config.BOOTSTRAP_ROUNDS,
    seed: int = config.SEED,
) -> PromotionDecision:
    """Decide whether ``challenger`` should replace ``champion`` on labelled hold-out scores."""
    y = np.asarray(y)
    champ = np.asarray(champion_proba, dtype="float64")
    chall = np.asarray(challenger_proba, dtype="float64")
    columns = segment_columns if segment_columns is not None else config.PROMOTION_SEGMENTS

    # One shared draws matrix scores both ΔROC-AUC and ΔPR-AUC (was two independent bootstraps).
    bands = evaluate.bootstrap_diff_ci(y, chall, champ, n_rounds=n_rounds, seed=seed, alpha=alpha)
    delta_roc = bands["roc"]
    delta_pr = bands["pr"]
    primary_passed = delta_roc["diff_lo"] > mde and delta_pr["diff_lo"] > mde

    guardrails = [
        _brier_guardrail(y, champ, chall, brier_tol),
        _segment_guardrail(y, champ, chall, segment_frame, columns, segment_tol, min_segment_rows),
    ]
    guardrails_passed = all(g.passed for g in guardrails)
    promote = primary_passed and guardrails_passed

    reasons: list[str] = []
    if not primary_passed:
        reasons.append(
            f"primary gate not met: ROC-AUC LB {delta_roc['diff_lo']:.4f}, "
            f"PR-AUC LB {delta_pr['diff_lo']:.4f} vs MDE {mde} (incumbent wins ties)"
        )
    for g in guardrails:
        if not g.passed:
            reasons.append(f"guardrail failed [{g.name}]: {g.detail}")
    if promote:
        reasons.append("promoted: both paired lower bounds clear the MDE and all guardrails pass")

    return PromotionDecision(promote, primary_passed, delta_roc, delta_pr, guardrails, reasons)
