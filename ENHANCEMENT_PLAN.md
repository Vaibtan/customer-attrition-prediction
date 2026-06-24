# Enhancement Plan — Churn Prediction as a Production Platform

> **Branch:** `enhancement` · **Date:** 2026-06-21 (rev. 2026-06-23, two Codex adversarial-review rounds folded in)
> **Goal:** Evolve the leak-free churn *model* into a full, Dockerized, production-grade
> ML *platform* — one coherent production story, built on the principle of
> **"the most optimal resource per need."**
>
> **Revision history:**
> - **R1** — first adversarial review found the "we prove event data lifts AUC" thesis
>   *circular*. Reframed from *proof about the real world* to *validation of a leak-free
>   measurement instrument against known ground truth*.
> - **R2** — second review confirmed all infra/sequencing objections fixed, and sharpened
>   the crux: negative controls are **necessary, not sufficient** for leakage (added a
>   leakage-sentinel suite); the anchor is a **distribution anchor, not a real-outcome
>   cohort** (three distinctly-named baselines); pre-registration must be **enforceable**
>   (lock file + CI guard); the promotion gate and drift rule must be **fully specified**
>   to match the code.

---

## 1. The thesis (a validated instrument, scoped honestly)

The original submission ends on an unprovable claim — "the ceiling is a *data* problem;
event-level signals would lift AUC past 0.64." **We cannot prove that without real event
data.** Injecting label-correlated events and measuring an AUC rise is circular.

So the platform is **the leak-free measurement instrument you would build to test that
hypothesis on real data — validated against known ground truth before trusting it:**

- **Positive control** — inject a known latent signal; a correct, leak-free pipeline
  *recovers* it (AUC rises on a frozen simulator). Validates the **engineering**, not the
  business case. This result is explicitly **synthetic**.
- **Negative controls** — label-shuffled events and a null (non-informative) stream
  produce **no lift**. These are **necessary sanity checks, not a sufficiency proof** of
  leak-freeness.
- **Leakage-sentinel suite (§4.7)** — the actual leakage guard: a taxonomy of known
  leakage paths, each with a test. Leak-freeness is argued from this suite *plus* the
  negative controls, never from the controls alone.
- **Enforceable lock (§4.6)** — the simulator spec is frozen by a hash-pinned lock file
  and a CI guard *before* any measurement. A miss against the pre-declared effect size is
  **reported as a null**, never retuned away.

The scoped claim, stated in every doc, with **no use of the word "prove" about real data
or about leak-freeness**:

> *"I built a leak-free event→feature→score→monitor instrument and validated it on a
> frozen synthetic world: it recovers injected signal (positive control), reports nothing
> when there is nothing (negative controls), and passes a leakage-sentinel suite. Pointing
> it at real event streams is the next step. I make no real-world AUC claim without real
> event data."*

The broker/online store never touch the **batch scoring decision** (§3).

---

## 2. The guiding principle (resolved by interview)

**Real tools for the platform; DIY for the ML reasoning** — and every infra tool must
*earn it* (pass a substantive bar, not "photogenic").

| Layer | Decision | Justification — every tool earns it |
|---|---|---|
| Event transport | **Redpanda** + `confluent-kafka` | Earns it via a systems demo: replay guarantees, throughput target, backpressure, failure recovery, late/out-of-order handling — with tests. Not used to justify the thesis. `kafka-python` deprecated. |
| Stream processing | **Quix Streams** (event-time windows) | Earns it via adversarial parity tests vs offline DuckDB PIT (late/duplicate/reordered/boundary events → identical features). |
| Feature store | **DIY** — Redis (online) + DuckDB/Parquet (offline) + ASOF PIT | Keeps the ASOF join reviewable; PIT-correctness test is a centerpiece. Feast adds infra without conceptual gain here. |
| Orchestration | **Dagster** (asset-based) | Earns it via partitioned backfills over the timeline, drift-gated conditional-retrain branching, retries, lineage. The lineage UI is a *secondary* tiebreaker. |
| Tracking + registry | **MLflow** | **Gated on Phase 4** (once retrains exist). Local registry carries Phases 1–3. `@champion`/`@challenger` aliases; Dagster logs to it. |
| Drift detection | **DIY** — *type-aware*: **numeric = PSI + KS ("both fire"); categorical = PSI + chi-square (/JSD)**; + a domain-classifier multivariate alarm (separate) | KS is undefined for categoricals (today `monitoring.py` sets categorical KS to `NaN` — to be fixed). Each family gets a valid test; the domain classifier catches joint/correlation drift the univariate tests miss. |
| Delayed-label perf | **DIY** — reimplemented **CBPE** (~50 LOC), with explicit assumptions + failure criteria | **Assumes** calibrated probabilities, covariate-shift-only (stable P(y\|X)), adequate batch size. **Validated** against known simulated regimes: tracks truth under covariate drift; **diverges under concept drift — the intended demonstration**, reported as a hard limitation with error bands. |
| Promotion gate | **DIY** — paired bootstrap **ΔROC-AUC *and* ΔPR-AUC** lower-bound gate + guardrails; incumbent wins ties | Promote on calibration + ranking, **not dollars**. Fully specified in the frozen policy (§4.8). EV is a **sensitivity analysis** across `COST_SCENARIOS`, never the gate. |
| Ops observability | **Prometheus + Grafana** | **Ops-only**: live FastAPI latency/throughput/errors, where pull-based scraping is correct. Explicitly *not* ML monitoring. |
| ML mission-control | **Streamlit** | Risk tiers, drift-over-time, perf estimates, promotion history incl. *rejected* challengers. |
| Everything | **Docker Compose** | One `docker compose up` brings the platform online. |

---

## 3. Product shape & scoring paths (resolved — R1 fixed a real contradiction)

The churn **decision is batch** (nightly): event log → **offline** PIT features (DuckDB)
→ batch score → risk tiers + reason codes. This is the production path.

One **scoped online path**, justified by real concerns:

1. **On-demand single-customer scoring** — FastAPI `/score` reads the latest **online**
   features from Redis (a CSM/UI pulling up one customer).
2. **Train/serve-consistency demonstration** — online features from Redis must **match the
   offline PIT features** for the same `(customer, t0)`; the parity test is the
   deliverable (the exact skew failure mode the platform exists to prevent).

The broker and online store **never** participate in the batch decision.

---

## 4. Population, simulator & the experimental protocol

### 4.1 Population (Superset) — the 1,600 are a *distribution anchor*, not a real-outcome cohort

The original **1,600 customers are a distribution/calibration anchor** embedded in a
larger generated population (target ~50k). Their role, stated honestly (R2):

- **Calibration target** — the synthetic population's static marginals + key joints are
  fitted to the real 1,600 so the simulation is production-shaped, not made-up.
- **Reference baseline** — `real_static_reference_auc ≈ 0.64` on the *real* labels marks
  "where the real problem sits." It is **reported on its own and never compared against
  event-augmented AUC** (that comparison would be apples-to-oranges; the event experiment
  uses simulated labels — §4.4).

The anchor does **not** provide real-outcome validation of event features. The event
experiment is explicitly **synthetic**.

### 4.2 Simulator (latent common-cause — fixes the fatal label→event leak)

Hand-rolled NumPy. A per-customer **latent health state** (continuous, drifts via a
stochastic process; seeded from static attributes — *never* the label) is the **common
cause** of both:

- **Events** — login inter-arrival lengthens, payment-failure probability rises, support
  sentiment trends negative, downgrades occur, as latent health falls.
- **The label** — churn in the label window is a hazard driven by the latent state.

The **churn label is never an input to event generation.** For the anchor, the latent
state is seeded from the *real static features* and the label is **re-simulated** — so the
anchor experiment is a synthetic *distribution-anchored* experiment (R2), not real-outcome
validation. We do not pretend otherwise.

### 4.3 Temporal contract (fixes S3)

Per row: **predict at `t0`; features from events in `(−∞, t0]`; label = churn in
`(t0, t0 + 90d]`.** Enforced by the PIT test `max(feature_ts) ≤ t0 < min(label_ts)`.

### 4.4 The three experiments + three distinctly-named baselines (fixes S2 + R2 ambiguity)

To stop conflating real-label and synthetic-label numbers (R2 New #2), **three baselines
are reported under fixed names, never substituted for one another:**

- `real_static_reference_auc` — static features, **real** labels, anchor. The ≈0.64
  reference. Reported alone.
- `synthetic_static_auc` — static features, **simulated** labels, anchor. (Sanity: should
  land near 0.64 if the synthetic world resembles reality.)
- `synthetic_static_plus_event_auc` — static **+ event** features, **simulated** labels,
  anchor. The positive-control headline.

The three experiments:

1. **Anchor instrument-validation (comparability).** Positive control = `synthetic_static_plus_event_auc`
   − `synthetic_static_auc` clears the **pre-registered minimum** on a paired bootstrap
   **ΔROC-AUC and ΔPR-AUC** CI (lower bound above the floor). **Negative controls**
   (label-shuffled events; null stream) show no lift. The leakage-sentinel suite (§4.7)
   must pass. Headline (scoped, synthetic) result.
2. **Synthetic-only (load/scale).** The ~50k population drives streaming
   throughput/replay/failure tests. **Not an evidence claim.**
3. **Blended (synthetic-domain only).** Reported strictly as a synthetic-domain result;
   cohort diagnostics (§4.5) **scope** it but do **not** make it externally valid (R2
   verdict 6) — we say so explicitly.

### 4.5 Cohort diagnostics (fixes S4)

Before any blended claim, report **side-by-side** anchor vs synthetic: base rate, feature
marginals + key joints, missingness, outliers, static-only AUC, calibration, score
distributions. These **scope** results to a domain; they are not a validity guarantee.

### 4.6 Pre-registration — enforceable lock (fixes T3 + R2 New #3)

`docs/simulator_spec.md` freezes the latent process, event-emission functions, label rule,
control definitions, and the **pre-declared minimum ΔROC-AUC and ΔPR-AUC** — before any
measurement. Enforcement (not just "committed and unchanged"):

- `simulator.lock.json` records: **sha256 of the spec + simulator config/params**, the
  **seed**, the **git SHA**, and the pre-registered effect floors + control definitions.
- The Phase-1 measurement entrypoint **recomputes the spec hash, asserts it equals the
  lock, and refuses to run on mismatch**; the lock is embedded in every result artifact.
- A **CI guard** fails any measurement run whose spec hash ≠ lock (or where spec + results
  changed in the same commit). Changing the experiment requires a deliberate,
  reviewable re-lock.

### 4.7 Leakage taxonomy & sentinel suite (R2 New #1 / B3 — the real leakage guard)

Negative controls are necessary but **not sufficient**. Leak-freeness is argued from a
sentinel test per known path:

| Leakage path | Sentinel test |
|---|---|
| Customer-ID memorization | `customer_id` excluded from features; permuting/removing it leaves scores unchanged; no ID-derived features |
| Train/test customer overlap | **group-aware split by `customer_id`** — a customer's multiple `(customer, t0)` rows never straddle the split |
| Post-`t0` feature leakage | the §4.3 PIT assertion on every row; fuzz event timestamps around `t0` |
| Timestamp-boundary bugs | strict inequality at `t0`; boundary-fuzz fixtures (events exactly at `t0`, `t0±ε`) |
| Preprocessing leakage | every transformer fit per-fold (extends the existing `test_pipeline_leakage`) over the new feature pipeline |
| Target-aware simulator tuning | enforced by the §4.6 lock: simulator params frozen before measurement |
| (Backstop) | label-shuffle + null-stream negative controls show no lift |

### 4.8 Promotion policy — fully specified, frozen (R2 New #4 / B5)

Locked in the frozen policy (and matched in code):

- **Primary gate** — paired bootstrap **ΔROC-AUC and ΔPR-AUC** at fixed **α = 0.05**
  (95% CI); challenger promoted only if **both** lower bounds exceed a **minimum
  detectable effect** floor (not merely > 0).
- **Guardrails** — no calibration regression (Brier within tolerance); **no per-segment
  degradation** on declared segments (`subscription_plan` tiers, `region`).
- **Tie policy** — **incumbent wins ties** (challenger must clear the MDE margin).
- **Implementation note** — `evaluate.py` currently has only `bootstrap_auc_diff_ci`
  (ROC-AUC). A **paired PR-AUC bootstrap** must be added (Phase 4).
- EV across `COST_SCENARIOS` is reported as sensitivity, never a gate.

---

## 5. Target architecture (data flow)

```
                              ┌──────────────────────────── Docker Compose ────────────────────────────┐
  simulator (NumPy)           │                                                                          │
  latent state → events       │   ┌─ BATCH (production decision) ──────────────────────────────────┐    │
  + label (common cause)  ──► │   │ event log (DuckDB/Parquet) ─► offline PIT features (ASOF) ─►    │    │
  superset pop, frozen+locked │   │   train ─► MLflow (Phase 4+) ─► batch score ─► tiers + reasons  │    │
                              │   └────────────────────────────────────────────────────────────────┘    │
                              │       │ events also ─►                                                    │
                              │   Redpanda ─► Quix Streams ─► Redis (ONLINE features) ─► FastAPI /score   │
                              │       (systems demo:        (parity vs offline PIT)        (on-demand)    │
                              │        replay/late/fail)                │                        │        │
                              │                                         └─ Prometheus ─► Grafana (OPS)    │
                              │   drift sim + type-aware detectors + CBPE ─► retrain ─► ΔAUC/ΔPR-AUC gate │
                              │       │                                              │ (+guardrails;      │
                              │       └────────────► metrics store ◄─────────────────┘  EV=sensitivity)  │
                              │                          │                                                │
                              │                          ▼                                                │
                              │                 Streamlit ML mission-control                              │
                              └──────────────────────────────────────────────────────────────────────────┘
              Dagster orchestrates batch/retrain assets (partitioned over the timeline, drift-gated branch)
              backtest/replay harness drives the whole timeline → centerpiece chart
```

---

## 6. Repo layout (proposed)

```
src/churn/                  # core library (extended)
  simulator/                # latent-state population + event generator + drift injection
  featurestore/             # online (Redis) + offline (DuckDB) + PIT joins + parity
  streaming/                # producer + Quix Streams aggregation logic
  drift/                    # type-aware PSI/KS/chi-square + domain-classifier + CBPE
  lifecycle/                # retrain + ΔAUC/ΔPR-AUC + guardrail promotion gate
  backtest/                 # replay harness
  (existing: config, data, cleaning, features, pipeline, evaluate, interpret,
             registry, scoring, train, plots, monitoring)
services/                   # thin deployment entrypoints: producer/ consumer/ api/ dashboard/
orchestration/              # Dagster defs/assets
infra/                      # Dockerfiles, prometheus.yml, grafana/ provisioning
docs/                       # simulator_spec.md (pre-registration), adr/
simulator.lock.json         # hash/seed/git-sha lock for the frozen spec
compose.yaml                # full stack
tests/                      # extended; leakage-sentinel suite; coverage gate stays >=80%
```

---

## 7. What makes this read as *senior* (post-review centerpieces)

1. **Validated instrument with a real leakage guard** — positive control (recovers signal)
   + negative controls (no false signal) + the **leakage-sentinel suite (§4.7)** + an
   **enforceable frozen spec (§4.6)**. We argue leak-freeness from the suite, never claim
   the controls "prove" it.
2. **Point-in-time correctness** — ASOF join + the `max(feature_ts) ≤ t0 < min(label_ts)`
   test + group-aware splitting. The leak-free obsession, extended into time.
3. **Train/serve parity under adversarial streams** — late/duplicate/reordered/boundary
   events produce identical online (Redis) and offline (DuckDB) features.
4. **Delayed-label estimation with stated limits** — CBPE under explicit assumptions;
   demonstrably blind to concept drift, reported as a limitation, validated against known
   regimes.
5. **Promotion with statistical teeth, no false precision** — paired ΔROC-AUC *and*
   ΔPR-AUC lower bounds past an MDE + calibration/segment guardrails, incumbent wins ties;
   EV is sensitivity. Proven by a test that a better-by-noise challenger is **not**
   promoted.

---

## 8. Phased build plan (thesis-first + thin slice early — resolved by interview)

Each phase ends **green** (tests + ruff pass; deliverable demonstrably works).

- **Phase 0 — Scaffolding + lock.** Repo layout, pyproject extras, Compose skeleton, CI
  scaffold; write **`docs/simulator_spec.md`**, generate **`simulator.lock.json`**, and
  wire the **spec-hash CI guard (§4.6)**. *Green:* `docker compose config` validates;
  existing 46 tests pass; spec + lock committed; CI guard active.

- **Phase 1 — Instrument validation, offline (NO infra).** Latent-state simulator →
  event log → DuckDB PIT features → the **leakage-sentinel suite (§4.7)** → the anchor
  experiment with **positive + negative controls** and paired **ΔROC-AUC/ΔPR-AUC** CIs,
  reporting the **three named baselines (§4.4)**. *Green:* sentinel suite passes; positive
  control clears the pre-registered floors **and** negative controls show no lift (or an
  honest null is recorded against the lock). Cohort diagnostics committed.

- **Phase 2 — Thin end-to-end vertical slice.** ONE event type through
  Redpanda → Quix → Redis → offline parity, on a minimal Dagster DAG. *Green:*
  **exhaustive** online/offline parity on a bounded replay + adversarial fixtures (late,
  duplicate, reordered, boundary).

- **Phase 3 — Full streaming + serving.** All event types; FastAPI `/score` reads Redis
  (scoped on-demand path); systems demo (replay, throughput, failure recovery). *Green:*
  full-population parity holds; failure/replay tests pass.

- **Phase 4 — Drift + delayed labels + retrain/promote + MLflow.** Drift simulation
  (covariate / prior / concept × sudden / gradual / recurring); **type-aware** detectors +
  CBPE; retrain loop; the **fully-specified promotion gate (§4.8)** incl. the new **paired
  PR-AUC bootstrap**; MLflow stood up now that retrains exist. *Green:* "better-by-noise →
  not promoted" test and "CBPE blind to concept drift" demo both pass; runs in MLflow with
  champion alias.

- **Phase 5 — Orchestration depth.** Dagster: partitioned backfills, schedules, drift-gated
  retrain branch, retries. *Green:* a partitioned backfill + a drift-triggered conditional
  retrain run end to end.

- **Phase 6 — Replay harness + observability.** Backtest replay over the drifting timeline
  → **centerpiece chart** (estimated vs true perf, drift alerts, retrain & recovery
  markers); Prometheus+Grafana (ops) + Streamlit (ML). *Green:* chart generated; dashboards
  live.

- **Phase 7 — Docs, CI, polish.** Rewrite README/docs into the single production story
  (scoped, non-overclaiming language; no "prove" about real data or leak-freeness); ADRs;
  extend CI; final full-stack `docker compose up` smoke. *Green:* whole stack up, all gates
  pass.

---

## 9. Risk & de-risking

- **Circularity (was fatal) → resolved** by the instrument reframe, the label-independent
  latent simulator, the **distribution-anchor** relabel + three named baselines, and the
  enforceable lock. **No real-world AUC claim.**
- **Leakage → guarded, not "proved"** by the sentinel suite (§4.7) plus negative controls.
  The plan never claims the controls alone prove leak-freeness (R2).
- **Thesis/null risk** — Phase 1 validates signal recovery before any container; nulls are
  reported against the lock, not retuned away.
- **Integration risk** — the Phase 2 thin vertical slice surfaces streaming/serialization/
  boundary bugs early.
- **Cargo-cult risk** — every infra tool passes its "earn it" bar (§2).

---

## 10. Carried-forward constraints

- **uv only — never pip.** Keep `print()` ASCII (Windows cp1252).
- Coverage gate **>=80%**; ruff check + format clean.
- The locked modelling findings stand: balanced target, weak signal, no SMOTE, tie-aware
  selection. **No real-world AUC claim without real event data**; the synthetic lift is
  scoped to instrument validation only.
- Ask before pushing / opening PRs.

---

## 11. Implementation items surfaced by review (tracked so none are lost)

- [ ] `evaluate.py`: add a **paired PR-AUC bootstrap** (only ROC-AUC `bootstrap_auc_diff_ci` exists) — Phase 4.
- [ ] `monitoring.py`: drift is **type-aware** — categorical uses PSI + chi-square/JSD, not KS (categorical KS is `NaN` today) — Phase 4.
- [ ] `simulator.lock.json` + spec-hash **CI guard** — Phase 0.
- [ ] **Leakage-sentinel suite** (§4.7), incl. group-aware split by `customer_id` — Phase 1.
- [ ] Three named baselines emitted as distinct metrics (`real_static_reference_auc`, `synthetic_static_auc`, `synthetic_static_plus_event_auc`) — Phase 1.
- [ ] CBPE: document assumptions + failure criteria; validate against known regimes — Phase 4.
```
