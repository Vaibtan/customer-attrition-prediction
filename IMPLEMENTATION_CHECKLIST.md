# Implementation Checklist — Churn Production Platform

> **Tracking companion.** The *what*, checkable and phase-ordered. The *why* lives in
> `ENHANCEMENT_PLAN.md` (design + architecture) and `PHASE0_LOCK_DECISIONS.md` (the decision
> record); `§`/`D` refs point there and rationale is **never restated here**. Phase order is
> load-bearing (thesis-first: kill the premise offline before building containers) — do not reorder.
>
> Branch: `enhancement`. Each phase ends **green** (tests + ruff pass; deliverable works).

---

## Always-on guardrails (every slice)

- [ ] uv only — never pip. `print()` stays ASCII (Windows cp1252).
- [ ] Coverage **>=80%**; `ruff check .` + `ruff format --check .` clean.
- [ ] No real-world AUC claim without real event data; no "prove" about real data or leak-freeness (§1).
- [ ] Every infra tool earns a substantive test bar, not "photogenic" (§2).

---

## Phase 0 — Scaffolding + Stage-1 lock — **DONE** (committed `6e9111b`, pushed)

> Stage-1 lock only (§4.6 / D1–D5). Froze the world + protocol + floor-derivation/oracle method.
> Analysis spec, analysis-code hash, and numeric floors are Stage 2 (end of Phase 1).

- [x] Repo layout (§6): `src/churn/{simulator,featurestore,streaming,drift,lifecycle,backtest}/`;
      top-level `services/`, `orchestration/`, `infra/`, `docs/`, `docs/adr/`.
- [x] pyproject extras: `streaming`, `featurestore`, `orchestration`, `tracking`, `dashboard`
      (base install stays lean).
- [x] `compose.yaml` skeleton: redpanda, redis, api, prometheus, grafana, dashboard, dagster —
      `docker compose config` validates.
- [x] CI scaffold: `.github/workflows/ci.yml` (`test`, `lock`, `lock-separation`, `docker`);
      checkout `fetch-depth: 0` (Layer-2 needs the merge-base).
- [x] `docs/simulator_spec.md` (D3, D5) — the frozen world: latent process, event/hazard forms,
      label rule, temporal contract, population + anchor, control defs, evaluation protocol,
      oracle/floor-derivation method.
- [x] `simulator.params.json` (D1): every numeric constant + selectors + control defs +
      `floor_design`. No confirmatory seed in the clear.
- [x] Golden-vector conformance tests (D1): `test_simulator_kernels.py`, `test_simulator_params.py`.
- [x] `simulator.lock.json` (D1): sha256 of spec + params + `kernels.py`/`params.py`/`beacon.py`,
      raw anchor hash, beacon recipe, control defs. Stage-2 fields null.
- [x] `check_lock` (D2): `python -m churn.simulator.lock`; Layer-1 CI job + runtime assert.
- [x] Confirmatory-seed isolation (D3): drand beacon recipe (chain hash + genesis + period), KDF,
      round rule `R = first round at time >= T_trusted + Δ`; CI enforcement.
- [x] CI Layer 1 (`lock`, consistency) + Layer 2 (`lock-separation`, same-commit separation).
- [ ] **Governance (ADR 0001):** `main` branch protection + required review — **manual GitHub
      setting, still open**.

**GREEN (met):** `docker compose config` validates · suite passes (93 tests) · spec + params +
lock + golden vectors committed · Layer-1/2 guards active. **Open:** branch protection (manual).

---

## Phase 1 — Instrument validation, offline (NO infra)

> **D7 re-lock (2026-07-02).** Phase-1 exploration hit the D5.9 recovery-feasibility risk; the
> pre-committed re-lock (κ 0.2→0.05 + stronger event coeffs, proxy quality only, ceiling verified
> unchanged) landed and was adversarially reviewed **SOUND** with a pre-committed one-shot bound
> (D7.1). See `PHASE0_LOCK_DECISIONS.md` D6–D7.

- [x] `simulator/` generator (§4.2, D5): latent-health common cause (seeded from static, **never the
      label**) + event generators on the frozen kernels — `rng/latent/population/events/generate.py`.
- [x] Superset population (§4.1, D5.7): ~50k synthetic, 1,600 real as distribution anchor;
      re-simulate the anchor label (`population.py`, D6.4).
- [x] Offline store: event log → DuckDB/Parquet (`featurestore/offline.py`, D6.3; pyarrow engine).
- [x] Offline PIT features via ASOF join (`featurestore/` offline half);
      `max(feature_ts) <= t0 < min(label_ts)` (§4.3) — strict PIT + D6.6 empty-window sentinels.
- [x] Leakage-sentinel suite (§4.7) — one test per path (`tests/test_leakage_sentinel.py`):
  - [x] `customer_id` excluded; permute/remove leaves scores unchanged
  - [x] group-aware split by `customer_id`
  - [x] post-`t0` PIT assertion per row + event-timestamp fuzz around `t0`
  - [x] timestamp-boundary fixtures (`t0`, `t0±ε`); strict inequality at `t0`
  - [x] preprocessing fit per-fold
  - [x] target-aware simulator tuning guarded by the lock (`check_lock` clean)
  - [x] backstop: label-shuffle + null-stream negative controls → no lift
- [x] Three named baselines as distinct metrics (§4.4): `real_static_reference_auc`,
      `synthetic_static_auc`, `synthetic_static_plus_event_auc` (`instrument/experiment.py`).
- [x] Develop on exploratory seeds only (D3); the confirmatory seed stays untouched.
- [x] Cohort diagnostics (§4.5): side-by-side anchor vs synthetic (`instrument/diagnostics.py`).

### End of Phase 1 — Stage-2 lock, then the single confirmatory run

- [x] Freeze `analysis_spec.json` (§4.6, D3): feature defs, model + hyperparams, preprocessing,
      selection metric, bootstrap unit = customer + method, CI method, oracle inputs/score/`N_oracle`,
      key dependency versions.
- [x] Analysis-code hash (D1/D2): hash the generator/feature/model/eval/measurement modules; activate
      the Layer-2 ANALYSIS set; extend `simulator.lock.json` (`lock.py --stage2`).
- [x] After beacon `R` emits: run the precommitted floor recipe
      (`churn.instrument.floors:compute_floors`) → MDE + oracle ceiling → floor =
      `max(MDE, 0.5·recoverable_lift)` per metric (ROC-AUC and PR-AUC) (`instrument/floors.py`).
- [x] Confirmatory run (single shot, D3): positive control clears **both** floors on the paired
      ΔROC-AUC / ΔPR-AUC lower bound (`instrument/measure.py`; real post-freeze drand round).
- [x] Measurement entrypoint (§4.6): recompute hashes, assert `== lock`, embed the lock in artifacts.
- [x] Tamper-evident results (D2): emit `reports/instrument_validation/**` (lock hash, code hash, git
      SHA, clean-tree marker, seeds, raw predictions + `predictions_sha256`); CI regenerates + compares.
- [x] Strict stopping rule (D3): a miss is a recorded null; per D7.1 no further recovery-feasibility
      re-lock is permitted.

**GREEN:** sentinel suite passes · Stage-2 lock committed · the single confirmatory run clears both
floors with tamper-evident results (or an honest null is recorded) · negative controls show no lift ·
cohort diagnostics committed.

---

## Phase 2 — Thin end-to-end vertical slice — **DONE** (real infra, committed)

> **Real-infra decision (2026-07-02).** Parity is proven through the ACTUAL deployment path against
> LIVE Redpanda + Redis (not mocks): a Dockerised `test-runner` (`infra/Dockerfile.test`, all
> extras) runs `pytest -m integration` networked to the compose `redpanda`/`redis` services. The
> host tier keeps the fast in-process parity + a `DictBackend` fake for unit speed; the container
> tier is the canonical guarantee. Run: `docker compose --profile test run --rm test-runner`.

- [x] Full 14-feature vector across all event types (exceeds the "one event type" thin slice) — the
      frozen `offline.FEATURE_COLUMNS` parity contract.
- [x] `streaming/producer.py` (`confluent-kafka`) → Redpanda topic; pure wire codec (int64-ns event
      time, NaN→null) host-tested, keyed by `customer_id`.
- [x] Quix Streams consumer (`services/consumer/app.py`): per-customer stateful event-time
      aggregation → the online reference reducer (`streaming/aggregate.feature_vector`).
- [x] Redis online store (`featurestore/online.py`): latest vector keyed by `customer_id`.
- [x] Online/offline parity: Redis feature == offline DuckDB PIT feature for `(customer, t0)` —
      **proven live** in `tests/test_streaming_integration.py` (produce→consume→Redis→compare).
- [x] Adversarial delivery: reordered + duplicated over the wire → byte-identical features (live);
      late/post-t0/boundary/empty covered by the in-process parity suite.
- [x] Minimal Dagster DAG (`orchestration/{assets,definitions}.py`): generate → offline PIT →
      online → `parity_report` (fails the run on any diff); materialises in-process.
- [x] `compose.yaml`: `topic-init` + `producer` + `consumer` services (streaming profile) +
      `test-runner` (test profile); `docker compose config` valid.

**GREEN (met):** live online/offline parity on a bounded adversarial replay through real
Redpanda+Redis passes · Dagster slice materialises + parity gate holds · host suite + ruff clean.

---

## Phase 3 — Full streaming + serving

- [ ] All event types through the streaming pipeline.
- [ ] FastAPI `/score` reads Redis online features (scoped on-demand path, §3).
- [ ] Systems demo (Redpanda's "earn it" bar, §4.4): replay guarantees, throughput target,
      backpressure, failure recovery, late/out-of-order handling — with tests.

**GREEN:** full-population parity holds · failure/replay tests pass.

---

## Phase 4 — Drift + delayed labels + retrain/promote + MLflow

- [ ] Drift simulation: covariate / prior / concept × sudden / gradual / recurring.
- [ ] `drift/` type-aware detectors: numeric = PSI + KS ("both fire"); categorical = PSI +
      chi-square/JSD (fix `monitoring.py` categorical KS = `NaN`); + domain-classifier alarm.
- [ ] CBPE (~50 LOC): explicit assumptions + failure criteria; validate against known regimes;
      demonstrate blind-under-concept-drift with error bands.
- [ ] `lifecycle/` retrain loop.
- [ ] `evaluate.py`: add paired PR-AUC bootstrap (only `bootstrap_auc_diff_ci` ROC-AUC exists today).
- [ ] Promotion gate (§4.8): paired ΔROC-AUC and ΔPR-AUC lower bounds past an MDE at α=0.05;
      guardrails (Brier within tolerance; no per-segment degradation on `subscription_plan`,
      `region`); incumbent wins ties; EV across `COST_SCENARIOS` = sensitivity, not gate.
- [ ] MLflow stood up (gated to here): `@champion`/`@challenger` aliases; Dagster logs to it.

**GREEN:** "better-by-noise → not promoted" test passes · "CBPE blind to concept drift" demo passes ·
runs in MLflow with champion alias.

---

## Phase 5 — Orchestration depth

- [ ] Dagster: partitioned backfills over the timeline, schedules, drift-gated conditional-retrain
      branch, retries, lineage.

**GREEN:** a partitioned backfill + a drift-triggered conditional retrain run end to end.

---

## Phase 6 — Replay harness + observability

- [ ] `backtest/` replay harness driving the whole drifting timeline.
- [ ] Centerpiece chart: estimated vs true perf, drift alerts, retrain & recovery markers.
- [ ] Prometheus + Grafana (ops-only: FastAPI latency/throughput/errors).
- [ ] Streamlit ML mission-control: risk tiers, drift-over-time, perf estimates, promotion history
      incl. rejected challengers.

**GREEN:** chart generated · dashboards live.

---

## Phase 7 — Docs, CI, polish

- [ ] Rewrite README/docs into the single production story — scoped, non-overclaiming language.
- [ ] ADRs in `docs/adr/`.
- [ ] Extend CI for the full stack.
- [ ] Final full-stack `docker compose up` smoke.

**GREEN:** whole stack up · all gates pass.

---

For the experimental-protocol reporting rules (three experiments) and the five senior centerpieces
this platform must demonstrate, see `ENHANCEMENT_PLAN.md` §4.4–§4.5 and §7 — not restated here.
