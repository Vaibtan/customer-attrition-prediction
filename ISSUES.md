# Issues Backlog — Churn Platform Cleanup

> **Provenance:** produced by a `/simplify` multi-agent review on branch `enhancement`
> (2026-07-04). 11 read-only reviewers across 4 lenses (concurrency / design / quality /
> efficiency), each finding adversarially verified. 51 findings → **4 applied**, **1 skipped**,
> **~18 distinct issues** below (deduped from 44 surface findings).
>
> This doc is **self-contained** so the session can be `/clear`ed and the work picked up cold.

---

## 0. Session state — what is ALREADY DONE (do not redo)

> **Update — session 2 (2026-07-05): P0 + P1 RESOLVED & COMMITTED.**
> - `70e6b60` — the 4 mechanical `/simplify` fixes below (committed as their own baseline commit).
> - `73cfde1` — **ISS-09** (bind `aggregate.py` windows to `_W`) + **ISS-10** (delete dead
>   `feature_vector_incremental`, hoist the double `pd.Timestamp`). Parity suite 7/7 green.
> - `f2281c0` — **ISS-01 (P0)**: `use_changelog_topics=True` default on `run_consumer` + persistent
>   `consumer-state` volume in compose + the wipe-state regression test
>   (`test_state_wipe_recovers_only_with_changelog`). **Verified on live Redpanda/Redis**:
>   `docker compose --profile test run --rm test-runner` → 7 passed (incl. all 4 streaming-systems
>   tests), the new test red-without / green-with the fix.
>
> Verification method for ISS-01 confirmed against current Quix Streams docs: checkpoint order is
> *produce-to-changelog → commit offsets → flush local state*, so the changelog (written before the
> offset commit) is what makes recovery correct even with an ephemeral `state_dir`.
>
> **Update — 2026-07-07: the whole P2 tier RESOLVED, COMMITTED & INTEGRATION-VERIFIED.**
> Design was grilled first (`grill-with-docs`) → ISS-04 chose full unification, recorded in
> **ADR 0005**. Commits: `77d8a85` ISS-06 (`OnlineStore.from_url`) · `2d38543` ISS-07/08 (lifecycle
> `ColumnSubsetModel` + `linear_boundary_xy`) · `67b61c1` ISS-03 (`aligned_level_counts`) · `b747869`
> ISS-05 (`make_cohort`, 12 sites) · `b4af59e` ISS-02/13 (shared `_bootstrap_metrics` + combined
> `bootstrap_diff_ci`) · `9878285` ISS-04 (unified `api.scoring_app` + real online `run_id`).
> **Verified on live infra**: `docker compose --profile test run --rm test-runner` → **7 passed**
> (incl. `test_serving_integration` asserting the online `run_id` end-to-end). Full non-integration
> suite green; ruff clean. Two backlog nuances corrected: ISS-07 docstrings also differed; ISS-08 is a
> **2-way** dup (`timeline.py` ≡ `test_retrain.py`), not "×3" — `replay._window` left alone.
>
> **Remaining open (deferred by decision, not bugs):** ISS-11 (P3 config), ISS-12 (P4 cbpe O(n)),
> ISS-14 (P4 Redis write amp — **grill the semantic tradeoffs first**); ISS-15..18 (P5 locked, need a
> re-lock). All P0/P1/P2 commits pushed to `origin/enhancement`.

These 4 mechanical fixes are **applied to the working tree (uncommitted)** and left the suite green
(host `pytest` pass, `ruff check`/`format` clean, `docker compose config` valid):

| File | Change | Safety |
|---|---|---|
| `src/churn/streaming/producer.py` | Extracted `_dumps_wire()`; `serialize_event` + `produce_events` share the one wire-JSON codec | byte-identical |
| `src/churn/backtest/replay.py` | Hoisted `angle = _angle(step, cfg)` once per step (was computed twice) | same value |
| `src/churn/drift/detectors.py` | `_categorical_drift` calls `psi_from_counts()` on counts it already builds (dropped redundant `categorical_psi()`) | byte-identical PSI |
| `compose.yaml` | Replaced stale "PHASE 0 SKELETON / nothing wired up" header with the real profile map | comment only |

**Skipped (deliberately):** `replay.py:75` `detect_drift → domain_classifier_auc` swap. Correct
finding but it's a cold offline harness; the saved work is microseconds (the fix *keeps* the real
cost, the 5-fold CV refit) while `detect_drift(...).covariate_shift` reads clearer. Tactical
micro-opt with a clarity cost. Revisit only if desired — see also **ISS-12**.

> **First action next session:** decide whether to commit the 4 applied fixes (`git status` will
> show them), then start on **ISS-01**.

---

## 1. Hard constraints — READ BEFORE TOUCHING ANYTHING

1. **Locked / frozen set — NEVER edit without a deliberate re-lock.** A pre-registration lock hashes
   these; editing any of them changes `analysis_code_sha256` and **invalidates the tamper-evident
   confirmatory PASS** (`reports/instrument_validation/`). Per decision **D7.1** a re-lock is a spent
   one-shot needing a fresh beacon round. The locked modules:
   ```
   src/churn/simulator/{kernels,params,beacon,rng,latent,population,events,generate}.py
   src/churn/featurestore/offline.py
   src/churn/instrument/{model,experiment,floors,measure}.py
   simulator.params.json   docs/simulator_spec.md
   ```
   All **ISS-15..ISS-20** below live here → they are a **post-re-lock backlog**, deferred by design.
2. **Parity contract.** `src/churn/streaming/aggregate.py` (`CustomerAggregator`, `feature_vector`,
   `compute_features`) must stay **byte-identical** to the offline DuckDB PIT output for the same
   `(customer, t0)`. Guarded by `tests/test_streaming_parity.py` + `tests/test_streaming_integration.py`.
   Any change to that module must keep those green.
3. **Green bar every slice:** `uv run pytest -q`; `uv run ruff check . && uv run ruff format --check .`;
   coverage **≥80%**; `print()` stays **ASCII**; **uv only, never pip**.
4. **Base modules outside the enhancement diff** (`pipeline.py`, `train.py`, `registry.py`,
   `scoring.py`, `cleaning.py`, `data.py`, `features.py`, `interpret.py`, `plots.py`): fixes here are
   "outside the reviewed diff" — treat as separate/optional.

**Scope assumption recorded:** review scope = `main...HEAD` (≈ the whole platform); `pipeline.py`/
`train.py` reviewed read-only; locked set reviewed but never edited.

---

## 2. Concurrency summary (the explicit ask)

**No state-corrupting data races in owned code.** The in-process paths hold up: FastAPI runs a
**single uvicorn worker** (model loaded once, read-only after); `serve.py`'s `@lru_cache` and
`prometheus_client` counters are **internally locked**; the online lazy-init and joblib nesting are
wasteful, not corrupting. The **one real concurrency-correctness gap is a durability / exactly-once
issue** (**ISS-01**), not a race.

---

## Priority legend
`P0` correctness now · `P1` low-risk high-value (recommended next) · `P2` strategic/long-lasting ·
`P3` config hygiene · `P4` efficiency (optional) · `P5` locked — needs a re-lock decision.

---

## ISS-01 — [P0] Consumer state durability silently breaks parity after a restart
- **Files:** `services/consumer/app.py:72` (+ `compose.yaml` consumer service, ~L76–94)
- **Lens/verdict:** concurrency · CONFIRMED · editable
- **What:** `use_changelog_topics=False` **and** `CHURN_STATE_DIR=/tmp/quix-state` is unmounted.
  Quix commits Kafka offsets *before* flushing state; with no changelog and ephemeral state, **any
  ordinary container restart (or a crash in the commit→flush window) advances committed offsets while
  the per-customer aggregator + dedup state are lost.** Skipped events are **not** redelivered
  (`auto_offset_reset=earliest` only applies with no committed offset), so `CustomerAggregator`
  permanently undercounts and Redis diverges from the offline PIT vector — the **headline train/serve
  parity guarantee fails after a routine restart, with no recovery path**.
- **Why CI misses it:** `tests/test_streaming_systems.py::test_crash_recovery_converges` reuses the
  **same** `state_dir` across its simulated crash, so state always survives.
- **Fix:** set `use_changelog_topics=True` (state rebuilds from the durable changelog, produced before
  the offset commit) — this is the complete fix for the offset/state contract. Optionally also mount
  `CHURN_STATE_DIR` on a persistent named volume. Ensure changelog topics are auto-creatable on the
  Redpanda cluster and that the integration/parity test config exercises the deployed path.
- **New test (required):** a regression test that **wipes `state_dir`** (not just stops the consumer)
  between two `run_consumer` calls with the **same** `consumer_group` + committed offsets, asserting
  the online store matches the offline PIT vector only with the fix enabled.

---

## Strategic dedup cluster — [P2] "one mechanism, many copies"
> The deep, long-lasting refactors. Each is a concept reimplemented in N places that must be kept
> manually in sync. Do these **test-first, one PR-sized slice each**. All editable (non-locked).

### ISS-02 — Percentile bootstrap reimplemented ×4
- **Files:** `src/churn/evaluate.py:172` (`bootstrap_auc_ci`, `bootstrap_auc_diff_ci`,
  `bootstrap_pr_auc_diff_ci`) + `src/churn/drift/cbpe.py:55` (`estimate`)
- **Fix:** extract `_percentile_bootstrap(y_true, metric_fn, n_rounds, seed, alpha)` that builds the
  resample-index matrix, applies a passed metric callable per valid resample, and returns the
  percentile band. The 4 sites differ only in the scored metric. (See also **ISS-13** — the promotion
  gate can share the *same* draws matrix across the ROC + PR bootstraps.)

### ISS-03 — Categorical level-alignment reimplemented ×3
- **Files:** `src/churn/monitoring.py` (`categorical_psi` ~L27, `jensen_shannon_distance` ~L54),
  `src/churn/drift/detectors.py:72` (`_categorical_drift`)
- **Fix:** add `aligned_level_counts(ref, cur) -> (levels, ref_counts, cur_counts)` in `monitoring.py`
  (MISSING_TOKEN fill + sorted union + reindexed `value_counts`) and route all three through it.

### ISS-04 — Serve / online app scaffold divergence (folds in 2 more findings)
- **Files:** `api/online.py:56` vs `api/serve.py:50`
- **What:** `api/online.py` reimplements `serve.py`'s lazy-load+cache / health→degraded / score→503 /
  metrics scaffold with a **divergent** cache idiom (mutable `state` dict + `_scorer()` closure vs
  `@lru_cache`). **Folds in:** (a) `api/online.py:57` unlocked check-then-act lazy-init (concurrent
  first requests each reload the model — benign but wasteful); (b) `api/online.py:87`
  `OnlineScoreResponse.model_run_id` reports the configured `run_dir` path, **not** the real loaded
  `run_id` (inconsistent with `serve.py`).
- **Fix:** extract a shared `ModelHolder` / `scoring_app(loader)` scaffold in `api/`; have both apps
  build on it. Make `load_online_model` carry `registry.load_run(...).run_id` (currently dropped) and
  return it as `model_run_id`.

### ISS-05 — Single-`t0` cohort frame hardcoded ~10×
- **Files:** `orchestration/assets.py:55`, `orchestration/timeline.py:56`,
  `src/churn/serving/online_model.py:101`, + ~several tests
- **What:** `pd.DataFrame({"customer_id": ids, "t0": [t0]*len(ids)})` re-hardcoded everywhere.
- **Fix:** `make_cohort(ids, t0)` in a **NEW non-locked module** (e.g. `src/churn/featurestore/cohort.py`
  — **NOT** the locked `offline.py`); call it from all sites.

### ISS-06 — `OnlineStore(RedisBackend(url))` construction ×3
- **Files:** `api/online.py:53`, `services/consumer/app.py` (`build_online_store`), + tests
- **Fix:** add `OnlineStore.from_url(url)` (or `open_online_store(url)`) factory in
  `src/churn/featurestore/online.py`; call it from the sites.

### ISS-07 — Column-subset model adapter duplicated in prod + test
- **Files:** `orchestration/timeline.py:75` (`_SubsetModel`) == `tests/test_retrain.py:65` (`_Wrap`),
  byte-for-byte
- **Fix:** promote one `ColumnSubsetModel` adapter into `src/churn/lifecycle/` (next to
  `retrain_and_gate`/`evaluate_promotion`); import it in both places.

### ISS-08 — Synthetic gaussian + linear-boundary generator ×3
- **Files:** `src/churn/backtest/replay.py:48` (`_window`), `orchestration/timeline.py`
  (`_reference_and_current`), `tests/test_retrain.py`
- **Fix:** one `synthetic_classification(rng, n, dims, angle=0.0)` helper in a shared demo/fixtures
  util; build all three matrices through it.

---

## Parity module — [P1] editable, low-risk, RECOMMENDED NEXT
> `aggregate.py` is not locked but is parity-critical. Both are byte-identical / safe and guarded by
> `test_streaming_parity`. Recommended to apply early.

### ISS-09 — `CustomerAggregator.add` hardcodes window lengths (divergence landmine)
- **File:** `src/churn/streaming/aggregate.py:188`
- **What:** `add()` hardcodes `14/28/90/180/365` as magic numbers while the reference path reads them
  from `DEFAULT_FEATURE_SPEC["windows"]` via `_W`. The module's own "windows from a single source"
  claim is only half-true — **change a window in the spec and the O(1) fast path silently diverges
  from offline/reference.**
- **Fix:** bind `near=_W['near_days']`, `short=_W['short_days']`, `mid=_W['mid_days']`,
  `pay=_W['pay_days']`, `full=_W['full_days']` once (constants for a fixed `t0`); use in `add()`.
  Byte-identical; removes the landmine. **Recommend applying, guarded by the parity suite.**

### ISS-10 — `aggregate.py` dead code + micro-opt
- **File:** `src/churn/streaming/aggregate.py:247` (`feature_vector_incremental`) — **confirmed
  unreferenced** (no prod/test/ADR use). Delete, **or** keep as deliberate public API symmetry (design
  call — it mirrors `feature_vector`). Also `feature_vector` (~L144) builds `pd.Timestamp(ts)` twice
  per event → hoist once (walrus). Behavior-preserving; low value.

---

## ISS-11 — [P3] Config as single source of truth
- **Files:** `src/churn/drift/detectors.py:32` (`PSI_THRESHOLD = 0.2`) vs `monitoring.py`
  `build_drift_report` Alert Guide prose (`0.10`/`0.20`); `detectors.detect_drift:140` re-derives the
  numeric/categorical partition by **dtype-sniffing** instead of `config.BASE_NUMERIC`.
- **Fix:** centralize PSI band cutoffs in `config` (e.g. `PSI_WATCH=0.10`, `PSI_INVESTIGATE=0.20`,
  `PSI_THRESHOLD=0.20`) and reference from both; derive the default partition from
  `config.BASE_NUMERIC` / declared feature lists.

---

## Efficiency — [P4] optional (editable)
### ISS-12 — `cbpe.estimate` re-sorts every bootstrap draw
- **File:** `src/churn/drift/cbpe.py:55` — O(n_rounds · n log n): `estimate_auc(p[idx])` argsorts each
  of `n_rounds` draws. **Fix:** pre-sort `p` once; per round turn the index draw into per-sorted-
  position weights (`w = np.bincount(idx)[order]`) and evaluate a weighted soft-AUC in O(n). (Coordinate
  with **ISS-02**.)

### ISS-13 — Promotion runs two independent paired bootstraps
- **File:** `src/churn/lifecycle/promotion.py:104` — `bootstrap_auc_diff_ci` then
  `bootstrap_pr_auc_diff_ci` regenerate the **same** `(n_rounds × n)` draws matrix. **Fix:** one
  combined `evaluate.bootstrap_diff_ci` that generates draws once and computes both ROC + PR diffs in a
  single pass. (Natural companion to **ISS-02**.)

### ISS-14 — Redis write amplification
- **Files:** `src/churn/featurestore/online.py:59` (`put_many` loops `put` → N round-trips; add
  `set_many` via redis-py pipeline/MSET); `services/consumer/app.py:102` (a Redis SET per event when
  only the final vector is read — debounce, **changes freshness semantics**); `services/consumer/app.py:93`
  (per-event dedup markers `s:{eid}` never expire → **unbounded state growth**; bound with TTL/LRU sized
  to the redelivery window — load-bearing for idempotency, so a correctness-vs-memory tradeoff).

---

## Locked backlog — [P5] needs a re-lock decision (DO NOT touch while the lock holds)
> Surfaced for completeness. Each requires spending a re-lock (see constraint #1). **Zero urgency
> while frozen** — a frozen duplicate cannot drift.

### ISS-15 — [HIGH severity smell] Oracle ceiling duplicates the generative chain
- **File:** `src/churn/instrument/floors.py:79` (`_oracle_ceiling`) reimplements the
  `q → health-path → hazard_prob → Bernoulli-label` chain already in
  `simulator/generate.py::build_population_dataset`. If the generative model ever changes, the oracle
  and the labels **silently diverge**. **Fix (post-re-lock):** extract
  `simulator.generate.draw_oracle_labels(q, params, latent_rng, label_rng) -> (h_t0, oracle_score, y)`;
  call from both. *(Highest-severity smell in the codebase, but frozen — so no urgency now.)*

### ISS-16 — `offline.py` feature-spec duplicated (also PARITY)
- **File:** `src/churn/featurestore/offline.py:35` — window sizes live in `FEATURE_COLUMNS` name
  strings, `DEFAULT_FEATURE_SPEC.windows`, and the `_finalize` fillna enumeration. **Fix (post-re-lock,
  byte-identical PIT):** drive names, SQL FILTER windows, and `_finalize` from one feature-spec table.

### ISS-17 — `experiment.py` coupling / typing / perf (3 sub-findings)
- **File:** `src/churn/instrument/experiment.py` — `positive_control` (L101) couples baseline
  computation with the paired-bootstrap CI (forces `run_confirmatory` to reimplement it) → split into
  `positive_control_from_base(...)`; `compute_baselines` (L90) returns a dict conflating reported
  metrics with `_`-prefixed private prediction arrays → return a typed `Baselines` dataclass;
  `paired_diff_ci` (L51) rebuilds a per-round Python list of ~n_customers index arrays →
  vectorize (offsets/lengths/flat-rows + `np.add.reduceat`).

### ISS-18 — Scattered taxonomies + redundant wrappers
- `src/churn/simulator/events.py:33` — event-family taxonomy scattered across rng stream names, the
  events dicts, `params.json` events, and `null_stream.zeroed_b` → one event-family registry.
- `src/churn/simulator/generate.py:31` — static categorical trio + full static-feature list hardcoded
  in ≥5 modules → define `STATIC_CATEGORICAL`/`STATIC_FEATURES` once.
- `src/churn/instrument/measure.py:162` — `_roc`/`_ap` redundant metric wrappers with function-local
  imports → import at module top, delete the wrappers.

---

## Recommended sequence
1. ~~**ISS-01** (P0 correctness — the flagship guarantee) + its crash test.~~ ✅ done (`f2281c0`).
2. ~~**ISS-09 / ISS-10** (P1 parity-module, byte-identical, quick).~~ ✅ done (`73cfde1`).
3. ~~**ISS-02 … ISS-08** (P2 strategic dedup).~~ ✅ done (`77d8a85`, `2d38543`, `67b61c1`, `b747869`,
   `b4af59e`, `9878285`) + ADR 0005; 7-passed on live infra.
4. **ISS-11 … ISS-14** (P3/P4 config + efficiency). ← next. NB ISS-14 debounce/TTL change semantics
   (grill first).
5. **ISS-15 … ISS-18** — only if/when a re-lock is decided (they ride along on that commit).
