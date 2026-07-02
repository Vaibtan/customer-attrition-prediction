# ADR 0004 — Promotion policy: statistical teeth, no false precision

**Status:** Accepted (Phase 4, 2026-07-02) · **Context:** `ENHANCEMENT_PLAN.md` Sec 4.8

## Context

A retrain must not silently replace the champion on noise. "Better on the test set" is not enough:
sampling noise, a calibration regression, or a per-segment collapse can all masquerade as an
improvement. The gate must have statistical teeth without inventing precision the data cannot support.

## Decision

`lifecycle/promotion.evaluate_promotion` promotes a challenger only if ALL hold:

- **Primary gate:** the paired-bootstrap lower bounds of **both** ΔROC-AUC and ΔPR-AUC exceed a
  minimum detectable effect (`PROMOTION_MDE`), at α = 0.05 — not merely > 0. Same resamples score
  both models (properly paired). PR-AUC's paired bootstrap was added to `evaluate.py`.
- **Guardrails:** no calibration regression (challenger Brier within tolerance of the champion's);
  no per-segment ROC-AUC degradation beyond tolerance on the declared segments (`subscription_plan`,
  `region`).
- **Tie policy:** the **incumbent wins ties** — anything short of clearing the MDE margin AND passing
  every guardrail keeps the champion.
- Expected value across `COST_SCENARIOS` is reported as **sensitivity**, never a gate.

`lifecycle/retrain.retrain_and_gate` wraps this: retraining produces a challenger; only the gate
promotes it. Promotion is realised as an atomic MLflow alias move (`@champion` := challenger version).

## Consequences

- The centerpiece test proves a **better-by-noise challenger is not promoted**, and a genuine
  improvement is; a calibration regression and a segment collapse each block promotion.
- The gate is deterministic and auditable: every decision surfaces the two lower bounds + each
  guardrail's verdict.
- MDE/tolerances are policy parameters in `config.py`, tuned once and frozen with the policy.
