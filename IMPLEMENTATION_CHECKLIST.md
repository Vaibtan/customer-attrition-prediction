# Implementation Checklist — Churn Production Platform

> Coding-time companion to **`ENHANCEMENT_PLAN.md`** (the authoritative design).
> Section refs like **§4.7** point into that plan for the *why*; this doc is the *what*,
> flattened into checkable slices. Tick items as they land. Phase order is load-bearing
> (**thesis-first**: kill the premise offline before building containers) — do not reorder.
>
> Branch: `enhancement`. Each phase ends **green** (tests + ruff pass; deliverable works).

---

## Always-on guardrails (apply to every slice)

- [ ] **uv only — never pip.** `print()` stays ASCII (Windows cp1252).
- [ ] Coverage gate **>=80%**; `ruff check .` + `ruff format --check .` clean.
- [ ] **No real-world AUC claim** without real event data — synthetic lift is scoped to
      instrument validation only. No use of "prove" about real data or leak-freeness.
- [ ] Every infra tool must **earn it** (§2) — a substantive test bar, not "photogenic".
- [ ] **Ask before pushing / opening PRs.**

---

## Phase 0 — Scaffolding + lock

- [ ] **Repo layout** (§6): create package subdirs under `src/churn/` — `simulator/`,
      `featurestore/`, `streaming/`, `drift/`, `lifecycle/`, `backtest/` (each with
      `__init__.py`); top-level `services/`, `orchestration/`, `infra/`, `docs/`, `docs/adr/`.
- [ ] **pyproject extras**: optional-dependency groups mirroring the existing `serve` extra
      (e.g. `streaming` = confluent-kafka/quix-streams, `featurestore` = redis/duckdb,
      `orchestration` = dagster, `tracking` = mlflow, `dashboard` = streamlit). Base install
      stays lean.
- [ ] **`compose.yaml` skeleton**: service placeholders (redpanda, redis, prometheus,
      grafana, api, dashboard, dagster) — minimal but enough that `docker compose config`
      validates.
- [ ] **CI scaffold**: extend `.github/workflows/ci.yml` with the new job structure +
      the spec-hash guard job (below).
- [ ] **`docs/simulator_spec.md`** (§4.6) — write **and freeze** the pre-registration:
      latent process, event-emission functions, label rule (churn in `(t0, t0+90d]`),
      control definitions (label-shuffle, null-stream), and the **pre-declared minimum
      ΔROC-AUC and ΔPR-AUC** (the seed is recorded in the lock, not the spec).
- [ ] **`simulator.lock.json`**: sha256 of spec + simulator config/params, seed, git SHA,
      effect floors, control definitions. (A small hashing helper computes it — needed by the
      §4.6 guard, which *recomputes the spec hash*.)
- [ ] **Spec-hash CI guard** (§4.6): recompute spec hash, assert `== lock`, fail on mismatch
      (or when spec + results change in the same commit). Lock embeds into every result
      artifact (consumed in Phase 1).

**GREEN:** `docker compose config` validates · existing test suite passes · spec + lock
committed · CI guard active.

---

## Phase 1 — Instrument validation, offline (NO infra)

- [ ] **`simulator/`** (§4.2): latent-health-state generator as **common cause** (seeded from
      static attributes, **never the label**); event generator (login inter-arrival,
      payment-failure prob, support sentiment, downgrades). Drift injection is **Phase 4**, not here.
- [ ] **Superset population** (§4.1): ~50k synthetic with the **1,600 real as a
      distribution/calibration anchor** — fit synthetic marginals + key joints to the real
      1,600; **re-simulate the anchor label** (anchor experiment is explicitly synthetic).
- [ ] **Offline store**: event log -> DuckDB/Parquet.
- [ ] **Offline PIT features** via **ASOF join** (`featurestore/` offline half); temporal
      contract enforced: `max(feature_ts) <= t0 < min(label_ts)` (§4.3).
- [ ] **Leakage-sentinel suite** (§4.7) — one test per path:
  - [ ] `customer_id` excluded; permute/remove leaves scores unchanged; no ID-derived features
  - [ ] **group-aware split by `customer_id`** — a customer's rows never straddle the split
  - [ ] post-`t0` PIT assertion on every row + event-timestamp fuzz around `t0`
  - [ ] timestamp-boundary fixtures (events at `t0`, `t0±ε`); strict inequality at `t0`
  - [ ] preprocessing fit per-fold (extend `test_pipeline_leakage` over the new feature pipeline)
  - [ ] target-aware simulator tuning guarded by the §4.6 lock
  - [ ] backstop: label-shuffle + null-stream negative controls show no lift
- [ ] **Three named baselines** as distinct metrics (§4.4), **never substituted**:
      `real_static_reference_auc` (static features, **real** labels — reported **alone**, never
      vs event-AUC); `synthetic_static_auc` (static, **simulated** labels); and
      `synthetic_static_plus_event_auc` (static **+ event**, **simulated** labels — the
      positive-control headline).
- [ ] **Positive control**: `synthetic_static_plus_event_auc − synthetic_static_auc` clears
      the pre-registered floor on a paired bootstrap **ΔROC-AUC and ΔPR-AUC** lower bound.
- [ ] **Negative controls**: label-shuffled events + null stream -> **no lift**.
- [ ] **Measurement entrypoint**: recompute spec hash, assert `== lock`, refuse on mismatch;
      embed lock in result artifacts; **record an honest null** if a floor is missed (never
      retune).
- [ ] **Cohort diagnostics** (§4.5): side-by-side anchor vs synthetic — base rate, marginals
      + key joints, missingness, outliers, static-only AUC, calibration, score distributions.
      These **scope** results to a domain; they are **not a validity guarantee**.

**GREEN:** sentinel suite passes · positive control clears the floors **and** negative
controls show no lift (or an honest null is recorded against the lock) · cohort diagnostics
committed.

---

## Phase 2 — Thin end-to-end vertical slice

- [ ] Pick **ONE event type** (e.g. login events).
- [ ] **`streaming/`**: producer (`confluent-kafka`) -> Redpanda topic.
- [ ] **Quix Streams** consumer: event-time windowed aggregation for that one feature.
- [ ] **Redis online store** (`featurestore/` online half): write latest feature vector keyed
      by `customer_id`.
- [ ] **Online/offline parity**: Redis online feature == offline DuckDB PIT feature for the
      same `(customer, t0)`.
- [ ] **Adversarial fixtures**: late / duplicate / reordered / boundary events -> identical
      features.
- [ ] **Minimal Dagster DAG** wiring the slice.

**GREEN:** exhaustive online/offline parity on a bounded replay **+** adversarial fixtures pass.

---

## Phase 3 — Full streaming + serving

- [ ] All event types through the streaming pipeline.
- [ ] **FastAPI `/score`** reads Redis online features (scoped on-demand path, §3).
- [ ] **Systems demo** (the "earn it" bar for Redpanda): replay guarantees, throughput
      target, backpressure, failure recovery, late/out-of-order handling — **with tests**.
      This is the *synthetic-only load/scale* experiment (§4.4) — **not an evidence claim**.

**GREEN:** full-population parity holds · failure/replay tests pass.

---

## Phase 4 — Drift + delayed labels + retrain/promote + MLflow

- [ ] **Drift simulation**: covariate / prior / concept × sudden / gradual / recurring.
- [ ] **`drift/` type-aware detectors**: numeric = PSI + KS ("both fire"); **categorical =
      PSI + chi-square/JSD** (fix `monitoring.py` categorical KS = `NaN` today); +
      domain-classifier multivariate alarm (separate).
- [ ] **CBPE** (~50 LOC): explicit assumptions (calibrated probs, covariate-shift-only,
      adequate batch) + failure criteria; validate against known regimes; demonstrate
      **blind-under-concept-drift** with error bands.
- [ ] **`lifecycle/` retrain loop**.
- [ ] **`evaluate.py`: add paired PR-AUC bootstrap** (only `bootstrap_auc_diff_ci` ROC-AUC
      exists today).
- [ ] **Promotion gate** (§4.8): paired **ΔROC-AUC and ΔPR-AUC** lower bounds past an **MDE**
      at α=0.05; guardrails (no calibration regression — **Brier within tolerance**; no
      per-segment degradation on `subscription_plan`, `region`); **incumbent wins ties**;
      EV across `COST_SCENARIOS` = **sensitivity, not gate**.
- [ ] **MLflow** stood up (gated to here): `@champion`/`@challenger` aliases; Dagster logs to it.

**GREEN:** "better-by-noise -> not promoted" test passes · "CBPE blind to concept drift"
demo passes · runs in MLflow with champion alias.

---

## Phase 5 — Orchestration depth

- [ ] **Dagster**: partitioned backfills over the timeline, schedules, drift-gated
      conditional-retrain branch, retries, lineage.

**GREEN:** a partitioned backfill **+** a drift-triggered conditional retrain run end to end.

---

## Phase 6 — Replay harness + observability

- [ ] **`backtest/` replay harness** driving the whole drifting timeline.
- [ ] **Centerpiece chart**: estimated vs true perf, drift alerts, retrain & recovery markers.
- [ ] **Prometheus + Grafana** (ops-only: FastAPI latency/throughput/errors).
- [ ] **Streamlit** ML mission-control: risk tiers, drift-over-time, perf estimates,
      promotion history incl. **rejected** challengers.

**GREEN:** chart generated · dashboards live.

---

## Phase 7 — Docs, CI, polish

- [ ] Rewrite README/docs into the single production story — scoped, non-overclaiming
      language (no "prove" about real data or leak-freeness).
- [ ] **ADRs** in `docs/adr/`.
- [ ] Extend CI for the full stack.
- [ ] Final full-stack `docker compose up` smoke.

**GREEN:** whole stack up · all gates pass.

---

## Experimental protocol & reporting rules (§4.4–§4.5)

The plan defines **three experiments**, each with a fixed reporting rule that must hold
wherever its results appear — never substitute one for another:

1. **Anchor instrument-validation** (Phase 1) — positive + negative controls, three named
   baselines. The scoped, **synthetic** headline result.
2. **Synthetic-only (load/scale)** (Phase 3) — the ~50k population drives
   throughput/replay/failure tests. **Not an evidence claim.**
3. **Blended (synthetic-domain only)** (surfaces in the backtest/dashboards + docs) —
   reported **strictly as a synthetic-domain result**; cohort diagnostics **scope** it but
   do **not** make it externally valid. We say so explicitly.

---

## Senior centerpieces (§7) — the five things this must demonstrate

1. Validated instrument with a real leakage guard (sentinel suite + pos/neg controls + lock).
2. Point-in-time correctness (ASOF + PIT assertion + group-aware split).
3. Train/serve parity under adversarial streams (online Redis == offline DuckDB).
4. Delayed-label estimation with stated limits (CBPE, blind to concept drift, error bands).
5. Promotion with statistical teeth, no false precision (ΔROC+ΔPR past MDE + guardrails).
