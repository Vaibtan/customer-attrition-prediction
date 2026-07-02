"""Retrain-and-gate loop -- fit a challenger, evaluate it against the champion, keep the winner.

Retraining is NOT promotion: a freshly trained challenger replaces the champion only if it clears
the promotion gate (Sec 4.8). Otherwise the incumbent is kept. This is the unit a drift-triggered
orchestration branch (Phase 5) calls when a detector fires.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from churn.lifecycle.promotion import PromotionDecision, evaluate_promotion


@dataclass(frozen=True)
class RetrainOutcome:
    challenger: object
    decision: PromotionDecision
    champion: object  # the model that remains champion after the gate (incumbent or challenger)
    promoted: bool


def _proba(model, x):
    return model.predict_proba(x)[:, 1]


def retrain_and_gate(
    train_fn: Callable[[], object],
    champion: object,
    x_holdout,
    y_holdout,
    segment_frame: pd.DataFrame | None = None,
    **gate_kwargs,
) -> RetrainOutcome:
    """Train a challenger via ``train_fn``; promote over ``champion`` only if the gate passes."""
    challenger = train_fn()
    decision = evaluate_promotion(
        y_holdout,
        _proba(champion, x_holdout),
        _proba(challenger, x_holdout),
        segment_frame,
        **gate_kwargs,
    )
    champion_after = challenger if decision.promote else champion
    return RetrainOutcome(challenger, decision, champion_after, decision.promote)
