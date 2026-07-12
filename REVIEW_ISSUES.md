# REVIEW_ISSUES — Full-Platform Review (2026-07-12)

> **Provenance:** independent full review of branch `enhancement` at `ae8baaa`, requested as a
> "review everything planned and built" pass. Method: 7 parallel read-only reviewers (simulator/lock,
> featurestore/streaming, serving/lifecycle, monitoring/orchestration, base ML, infra/CI/docs,
> architecture), every significant finding re-verified against the code by the coordinating session
> (several empirically — the pydantic `inf` acceptance and the two streaming divergences were
> reproduced, the Dagster partitioning wiring traced).
>
> **Baseline at review time:** host suite green (`uv run pytest -q -m "not integration"`, exit 0);
> `docker compose config` valid across all profiles; working tree clean.
>
> **Relationship to `ISSUES.md`:** zero overlap by construction — reviewers were given the open
> backlog (ISS-11/12/14, ISS-15..18) and told not to re-report it. Where a finding is *adjacent* to
> a tracked issue it is cross-referenced. IDs here are `REV-xx`.
>
> **Locked-set note:** findings marked **[LOCKED]** live in the pre-registration hash set (or their
> fix requires editing `lock.py`'s hash lists, which itself forces a re-lock). They are recorded for
> the post-re-lock backlog, same convention as ISS-15..18. None of them overturn the recorded
> confirmatory PASS (the substantive gate cleared its floor with real margin).

---

## 0. Resolution status (build session 2026-07-12)

> Design tier grilled first (`grill-with-docs`, all nine decisions → **ADR 0006/0007** +
> `CONTEXT.md`), then built in 8 commits, each green on the full host suite; live-infra sweep at
> the end of the session.
>
> **RESOLVED & COMMITTED:**
> - Quick wins: **REV-09** (inf 422s) · **REV-12** (route-template labels + uncaught-500
>   counting) · **REV-28** (locked cache) · **REV-15** (Dockerfile layers) · **REV-18** (MIT
>   LICENSE) · **REV-29** (.dockerignore) · **REV-23** (WRITEUP 11→10) · **REV-25** (stale local
>   runs pruned; finding corrected — `models/` was never git-tracked) · **REV-17** (ci.yml
>   marker filter, infra/README, MLflow pin sync-guard test, drift report regenerated).
> - Design slices: **A2/ISS-11** (`churn.settings.Settings` + PSI bands + declared-schema
>   partition) · **ADR 0007 / REV-04/05/11** (versioned envelope, strict 409 reader, NaN-proof
>   writes, aligned dedup + AVG semantics, new parity cases) · **ADR 0006 / REV-10/13/14**
>   (ChampionResolver TTL cutover + serve-last-good, shadow @challenger JSONL log + counter,
>   ok|degraded|stale health with reasons + version-skew warnings, `api-online:8001` service +
>   scrape target, log_model-returned version) · **REV-02** (label horizon + trigger-comparison
>   centerpiece + blindspot chart) · **REV-03** (cron → unpartitioned retrain job, tick test,
>   real Dagster service on 3001) · **REV-20** (st.cache_data).
>
> **STILL OPEN:** REV-16 (container/CI hardening bundle: non-root USER, restart policies,
> Grafana password, CI permissions/timeouts), REV-19/21/22/24/30 (deferred lows), the locked
> tier (REV-01/06/07/26/27 — rides the next re-lock), and architecture items 4 (alert rules +
> runbook), 6 (batch scoring as a scheduled asset), 7 (model card), 8 (authn/trust boundary).

---

## Verdict in one paragraph

The platform is fundamentally sound: no live correctness bug was found in a deployed happy path, the
leakage story holds up under tracing, the parity design is real (independent SQL vs Python
implementations + recompute oracle + live-infra adversarial streams), the promotion gate statistics
are conservative and correct, and the layering (`src/churn` never imports `api`/`services`/
`orchestration`) is genuinely clean. The issues found cluster in three bands: (1) **integrity/story
gaps** — the pre-registration perimeter has one unhashed dependency, and two centerpiece narratives
(delayed-label backtest, scheduled retrain) promise more than the code implements; (2) **latent
parity divergences** — two input invariants nobody validates, currently unreachable with simulator
data; (3) **production-hardening gaps** at the serving/observability boundary (inf inputs, blind
Redis trust, 500s invisible to Grafana, metrics cardinality). Nothing here is hard to fix except the
locked items.

---

## HIGH

### REV-01 — Lock integrity: `config.py` is an unhashed dependency of hashed analysis modules
- **Files:** `src/churn/simulator/lock.py:31-61` (hash lists) · `src/churn/simulator/population.py:23,36`
- **Category:** lock-integrity · **CONFIRMED** (verified directly) · fix requires re-lock ceremony
- **What:** `population.py` (in `ANALYSIS_MODULES`) reads `config.VALID_RANGES` to clip synthetic
  columns, but `config.py` is in neither `HASHED_FILES` nor `ANALYSIS_MODULES`. Editing
  `config.VALID_RANGES` moves the MDE floor pool (`build_population_dataset(n_synthetic=16000)`)
  with **zero lock drift** — defeating the "results cannot move without changing a hashed file"
  claim that the whole tamper-evidence story rests on.
- **Impact bound:** currently non-binding — `VALID_RANGES` only reaches the MDE floor, and the
  recorded PASS was decided by the substantive gate (0.081/0.098 floors vs ~0.11 lower bound), not
  the MDE (0.032/0.042). The *guarantee* is violated; the *verdict* is not.
- **Fix:** at the next deliberate re-lock, add `src/churn/config.py` to `ANALYSIS_MODULES` (or
  extract the constants the analysis path reads into an already-hashed module). Until then, note the
  limitation in `PHASE0_LOCK_DECISIONS.md` — an honest one-line disclosure is cheap and on-brand.

### REV-02 — Backtest sells a delayed-label regime it does not implement
- **Files:** `src/churn/backtest/replay.py:73,83-84` (+ docstring line 8, `services/dashboard/app.py:30` caption)
- **Category:** claim-vs-implementation / backtest realism · **CONFIRMED**
- **What:** the narrative (docstring, dashboard caption) is "predictions at t are scored against
  labels only available later." The code reads `true_auc` from the *same* window it scores and
  retrains on that *same* window's labels — label lag h = 0. Recovery therefore happens the very
  next step; a real delayed-label monitor reacts h steps late and retrains on a stale-but-labeled
  window. The centerpiece chart is more optimistic than the story it illustrates.
- **Fix:** add a `label_horizon_steps` to `ReplayConfig`; score step t, reveal its labels at t+h,
  gate retrain on the *revealed* set only. This makes the recovery lag visibly longer — which is
  the honest (and more impressive) chart.

### REV-03 — Weekly Dagster schedule cannot launch its partitioned job
- **Files:** `orchestration/definitions.py:37-41` · `orchestration/timeline.py:33,49`
- **Category:** orchestration wiring · **PLAUSIBLE** (API contract traced + checked against current
  Dagster docs; not executed against a live scheduler)
- **What:** `pit_snapshot` is `@asset(partitions_def=TIMELINE)` (StaticPartitionsDefinition);
  `timeline_backfill_job` selects it; `weekly_timeline_schedule` is a bare `ScheduleDefinition`
  around that job. A plain tick emits a RunRequest with no `partition_key`, which a partitioned job
  cannot launch — the first real `dagster dev` tick fails. The test
  (`test_orchestration_timeline.py:62-67`) asserts only the schedule's *name*, which is exactly what
  masks it.
- **Fix:** `build_schedule_from_partitioned_job(...)` (or a schedule function emitting
  `RunRequest(partition_key=...)` for the latest partition). Add a test that evaluates a tick.

---

## MEDIUM

### REV-04 — Latent parity divergence: null-valued `login`/`support` event poisons the online mean
- **Files:** `src/churn/streaming/aggregate.py:204-209,215` vs `src/churn/featurestore/offline.py` `AVG(...)`
- **Category:** parity (latent) · **CONFIRMED** (empirically reproduced) · offline side **[LOCKED]**
- **What:** offline `AVG(...)` skips SQL NULLs; both online paths do `sd28_sum += _as_float(value)`
  unconditionally, and `_as_float(None) = nan` — so one null-valued login makes
  `mean_session_depth_28d` `NaN` online while offline returns the mean of the rest. The wire format
  actively produces the input (`producer._clean_value` maps NaN→null). Unreachable with current
  simulator data (0 nulls in 642k logins / 43k supports) — guarded only by an **unvalidated
  invariant**.
- **Fix (non-locked side):** skip null values in the aggregator's mean accumulators (matches SQL
  semantics), or reject null-valued magnitude events at ingest with a schema check. Add a parity
  test with a null-valued login. Either keeps `offline.py` untouched.

### REV-05 — Dedup tie-break inconsistency: first-wins vs last-wins across the three implementations
- **Files:** `src/churn/featurestore/offline.py:69` (first) · `aggregate.py:72` incremental (first) ·
  `aggregate.py:305-307` `CustomerState.update` recompute reducer (**last**)
- **Category:** parity (latent) · **CONFIRMED** (empirically reproduced) · offline side **[LOCKED]**
- **What:** a redelivered duplicate `event_id` carrying a *different* payload is resolved
  first-wins by offline and the deployed incremental path, but last-wins by the recompute oracle —
  i.e. the parity suite's own oracle disagrees with what it certifies. Masked because every test
  duplicate is byte-identical.
- **Fix:** make `CustomerState.update` first-wins (`if event_id in self._events: return`) — it is
  the non-locked side and aligns all three; add a mutated-duplicate parity case. Also fix the
  `services/consumer/app.py:2-6` docstring, which describes the *reducer* as the deployed path when
  the code actually persists a `CustomerAggregator` (the docstring points readers at the wrong
  tie-break semantics).

### REV-06 — [LOCKED] Predictions-frame alignment is correct only by merge-order luck
- **Files:** `src/churn/instrument/measure.py:45-58`
- **Category:** correctness fragility · **CONFIRMED**
- **What:** the tamper-evidence frame pairs `customer_id` from `dataset.labels` with
  `y`/`p_static`/`p_event` from the design-merge order, ignoring the design's own `base["_groups"]`.
  Any customer dropped/reordered by the inner merges would silently mislabel rows *and then hash
  them*. Safe today only because n_synthetic=0 + LEFT-join PIT keeps all 1600 rows in population
  order. **Fix (post-re-lock):** take ids from `base["_groups"]`.

### REV-07 — [LOCKED] MDE floor is biased low (model-fit variance omitted)
- **Files:** `src/churn/instrument/floors.py:100-118`
- **Category:** methodology · **CONFIRMED** · non-binding in the recorded PASS
- **What:** the detectability floor subsamples n=1600 eval rows from a 16k pool, but the OOF
  predictions were fit once on the whole pool (~12.8k/fold) — the SE captures eval-set noise only,
  not the fit variance a real n=1600 run incurs (plus ~5% FPC from without-replacement draws). In a
  regime where the MDE gate binds, it would be too lenient. **Fix (post-re-lock):** refit per
  subsample, or state the floor as conditional-on-fitted-model in the spec.

### REV-08 — Confirmatory PASS ran on the manual round, bypassing the Δ=3600s anti-gaming buffer
- **Files:** `reports/instrument_validation/beacon_provenance.json` (self-disclosed)
- **Category:** provenance · **CONFIRMED**
- **What:** the headline PASS used the first drand round minutes after the freeze commit — not the
  canonical CI-enforced round R ≥ T_trusted + Δ that makes the seed un-gameable (D3). Honestly
  disclosed in the artifact, but the executed guarantee is weaker than the designed one; the
  un-gameable path exists only in prose.
- **Fix options:** (a) run the CI-canonical confirmatory once and record it alongside (strongest);
  (b) surface the caveat prominently in WRITEUP/PHASE0_LOCK rather than only inside the JSON.

### REV-09 — Serving accepts `inf` and returns saturated scores as authoritative
- **Files:** `api/serve.py:28-33` · `api/online.py:31-34` · (online path has no domain repair at all)
- **Category:** input validation · **CONFIRMED** (empirically: `1e400` → `inf` passes `Field(ge=0)`,
  NaN is rejected — inconsistent hardening)
- **What:** `inf ≥ 0` is true and `allow_inf_nan` is unset, so `{"account_age_days": 1e400}`
  reaches StandardScaler → LogisticRegression and returns probability 1.0/0.0 with HTTP 200. Batch
  is partially shielded (DomainRepair/Winsorizer) but unbounded-upper fields still pass inf through.
- **Fix:** `Field(ge=0, allow_inf_nan=False)` (pydantic v2) on all float fields in both request
  models + a contract test with `1e400`/`inf` payloads.

### REV-10 — `/health` degrades only on `FileNotFoundError`; every other load failure is an opaque 500
- **Files:** `api/scoring_app.py:44-63` · `src/churn/serving/online_model.py` (eager
  `metadata["tier_cutpoints"]` → `KeyError`) · `src/churn/registry.py:66-79,121-126`
- **Category:** error handling / health semantics · **CONFIRMED**
- **What:** corrupt pickle, sklearn version-skew, or a run missing `tier_cutpoints` raises through
  `get_model()`; `/health` 500s instead of reporting `degraded`, `/score` 500s hiding the cause.
  Routine MLOps failure modes surface as generic crashes. Related: `registry.save_run` records
  sklearn/pandas/numpy versions in metadata and **never checks them on load** — the guard data
  exists, unused.
- **Fix:** catch `Exception` (or a `ModelLoadError` wrapping the loader) in the scaffold's degraded
  path, include the reason in `/health`; add a version-mismatch warning on load. One place
  (`scoring_app.py`) fixes both apps — that's the point of ADR 0005.

### REV-11 — Online scorer trusts the Redis blob blindly: stale/partial vectors score silently
- **Files:** `src/churn/serving/online_model.py:52-61` · `src/churn/featurestore/online.py:62`
- **Category:** parity / freshness · **CONFIRMED** · read-side complement of ISS-14 (write-side)
- **What:** `assemble_row` does `event.get(c)` per feature; missing keys become NaN → median-imputed
  → a confidently-wrong "parity" score. No completeness check, no schema/version tag, no timestamp/
  freshness guard on the vector. Also `json.dumps(..., allow_nan=True)` would happily persist a
  literal `NaN` token (non-standard JSON; breaks non-Python readers) if REV-04 ever fires.
- **Fix:** stamp the payload with `feature_schema_version` + `as_of` at write; on read, 409/404 on
  version mismatch or incomplete vectors instead of imputing. This is also architecture item A3.

### REV-12 — Prometheus middleware: unbounded path-label cardinality AND uncounted 500s
- **Files:** `api/metrics.py:36-44`
- **Category:** observability · both **CONFIRMED**
- **What:** (a) raw `request.url.path` is a label — every scanner/typo URL mints a new timeseries
  (classic cardinality footgun); (b) an uncaught exception re-raises through `await call_next(...)`
  *before* `.inc()`, so genuine 500s never increment `churn_http_requests_total` — the Grafana
  "Error rate (5xx)" panel is blind to exactly the failures it exists for.
- **Fix:** label with the matched route template (`request.scope["route"].path`, constant bucket for
  unmatched); wrap `call_next` in try/except, count as 500, re-raise. Add a middleware test that
  throws from a route.

### REV-13 — MLflow: registered version recovered by `max(search_model_versions)` instead of the returned handle
- **Files:** `src/churn/lifecycle/mlflow_registry.py:44-46`
- **Category:** lifecycle correctness · **PLAUSIBLE** (assumption traced; not reproduced at scale)
- **What:** `log_model` already returns the `ModelVersion`; instead the code re-queries
  `search_model_versions` (a PagedList — first page under iteration, no guaranteed ordering) and
  takes `max(..., key=int)`. With many versions or a concurrent registration, the wrong version can
  be aliased `@champion` — the wrong model gets promoted. Tests register only 2 versions.
- **Fix:** use `mlflow.sklearn.log_model(...).registered_model_version`. Also: no test asserts the
  MLflow-loaded model *predicts identically* to the pre-log model (round-trip fidelity is proven for
  the file registry only).

### REV-14 — The headline online-parity endpoint is not deployable from compose
- **Files:** `compose.yaml:101-111` · `Dockerfile:14,29` · `infra/prometheus.yml:12-15`
- **Category:** compose / story-vs-deploy · **CONFIRMED**
- **What:** the `api` service runs `uvicorn api.serve:app` (batch `/score` only); no service serves
  `api/online.py`'s `/score/online`, and the root image installs only `--extra serve` (no redis
  client), so it couldn't. Consequences: `api`'s `depends_on: redis` is dead; the
  `online-scoring` service label can never exist in Prometheus/Grafana; and the README's
  train/serve-parity demo cannot be run via documented compose commands. Also cross-profile
  coupling is undocumented: `--profile observability` alone scrapes a nonexistent `api:8000`.
- **Fix:** add an `api-online` service (image with `serve`+`featurestore` extras, env `REDIS_URL`),
  scrape it, and note the serve+observability profile pairing in the README.

### REV-15 — Root Dockerfile busts its own dependency cache
- **Files:** `Dockerfile:5-14`
- **Category:** build hygiene · **CONFIRMED**
- **What:** `api/`, `data/`, `churn_prediction.py` are COPYed before `uv sync`, so any serving-code
  edit re-runs dependency resolution *and* the build-time `python -m churn.train`. The sibling infra
  Dockerfiles get the layering right; the root one doesn't.
- **Fix:** COPY `pyproject.toml uv.lock README.md src` → `uv sync` → COPY the rest.

### REV-16 — Container/CI hardening gaps (bundle)
- **Files:** all 5 Dockerfiles · `compose.yaml` · `.github/workflows/ci.yml:14-16,174-191`
- **Category:** ops hardening · **CONFIRMED**
- **What:** no image sets a non-root `USER`; no `restart:` policies or resource limits on
  long-running services; Grafana admin password is a literal `admin` in a committed file; CI grants
  `packages: write` at top level to every job; integration/compose-smoke jobs have no
  `timeout-minutes` (hang = 6h default). Individually small; together they read as "never operated."
- **Fix:** `USER` in images, `restart: unless-stopped` + basic limits, `GF_SECURITY_ADMIN_PASSWORD`
  from `.env`, job-scoped permissions, `timeout-minutes: 20` on docker jobs.

### REV-17 — Doc-vs-reality drift (bundle)
- **Files:** `docs/adr/0002-real-infra-test-strategy.md:17-18` vs `ci.yml:36` · `infra/README.md:6` ·
  `infra/Dockerfile.mlflow:6` vs `pyproject.toml:41` · `monitoring/drift_report.md:24-35`
- **Category:** docs honesty · **CONFIRMED**
- **What:** (a) ADR 0002 says the fast CI job runs `-m "not integration"`; ci.yml runs bare
  `pytest --cov` — the real guard is conftest's env-var skip, so leaked `REDIS_URL`-style envs would
  attempt live connections; (b) `infra/README.md` still says "Empty at Phase 0" over a fully
  populated tree; (c) MLflow *server* pinned 3.14.0 outside the lock while the client tracks
  `>=2.17` from `uv.lock` — two unrelated sources of truth; (d) the committed `drift_report.md`
  is from the old `psi|ks` schema the current code can no longer produce.
- **Fix:** add the marker filter to ci.yml (make the ADR true), refresh the two stale docs,
  regenerate the drift report, pin server+client from one place.

### REV-18 — No LICENSE file
- **Category:** portfolio hygiene · **CONFIRMED**
- A public portfolio repo without a license is a reviewer checkbox failure. Add MIT/Apache-2.0.

---

## LOW

- **REV-19** `src/churn/drift/cbpe.py:49-65` — the bootstrap band captures sampling variance only, so
  `concept_drift_suspected` also fires on plain *calibration failure*. Partially documented; add the
  caveat to the flag's docstring/dashboard caption. (Also: strict-below tie handling biases bootstrap
  draws ~O(1/n) low — negligible at n≥2500, noted for completeness.) **PLAUSIBLE**
- **REV-20** `services/dashboard/app.py:23`, `mission_control.py:20-22` — no `@st.cache_data`; every
  Streamlit rerun re-executes the full backtest (14×5-fold CV + 100-round CBPE) in the render path.
  **CONFIRMED**
- **REV-21** `src/churn/evaluate.py:117-151` — if *every* bootstrap resample is single-class,
  `_bootstrap_metrics` returns an empty array and `np.quantile([])` raises; the promotion gate should
  fail closed (no promotion), not throw. Unreachable on the current balanced holdout. **PLAUSIBLE**
- **REV-22** `src/churn/streaming/producer.py:83-92` — no delivery-report callback, no `BufferError`
  guard; the returned count is *enqueued*, not acknowledged. **CONFIRMED**
- **REV-23** `WRITEUP.md:64,127` — "11 customers with zero orders and positive spend" is actually 10
  (one of the 11 zero-order rows has NaN spend). One wrong number in a document whose brand is
  honesty. **CONFIRMED**
- **REV-24** `tests/test_pipeline_leakage.py:18-30` — the base-pipeline leakage sentinel checks only
  the scaler mean; a winsor-caps or imputer-median leak regression would stay green. Thinner than the
  "one test per leakage path" narrative. **CONFIRMED**
- **REV-25** `models/2026*` — stale LOCAL run dirs (correction: `models/` was never git-tracked —
  already ignored) whose metadata lists a feature (`logically_inconsistent`) that no longer exists;
  `load_run` on one would crash at scoring. Live path safe (mtime-latest run matches code).
  *Resolved: the four stale dirs pruned locally.* **CONFIRMED (downgraded)**
- **REV-26** [LOCKED] `src/churn/instrument/measure.py:138-148` — provenance records sklearn/numpy/
  pandas versions but not **duckdb**, which computes every PIT feature. **CONFIRMED**
- **REV-27** [LOCKED] `measure.py:61-67` — the 1e-10 predictions hash implies cross-platform
  reproducibility the beacon provenance itself scopes to *within-platform* (lbfgs/BLAS variance).
  Overclaim in the docstring, not a code bug. **PLAUSIBLE**
- **REV-28** `api/scoring_app.py:44-47` — the load-once cache is an unlocked check-then-act; two
  concurrent first requests both run `loader()`. Benign (last-write-wins, wasteful only) but ADR
  0005's Context dings `online.py` for this exact idiom, and the unified scaffold reproduces it.
  A `threading.Lock` is 3 lines. **CONFIRMED**
- **REV-29** `.dockerignore` — agent/tooling dirs (`.claude/`, `.codex/`) and `reports/` are sent to
  the build context; no image COPYes them (explicit COPY lists everywhere), but the `test-runner`
  bind-mounts the whole repo `:ro` into a root-running container. Add the entries. **CONFIRMED**
- **REV-30** `src/churn/train.py:114` — the reported winner CI is bootstrapped on *uncalibrated*
  probabilities while the deployed winner is calibrated. Numerically identical for AUC (monotone
  sigmoid); recorded as INFO, not a defect. **CONFIRMED**

### Test gaps worth closing (cross-cutting)
No test covers: concurrent first requests (REV-28); inf/out-of-range payloads (REV-09); corrupt /
missing-cutpoints / version-skew artifacts → degraded (REV-10); stale or partial Redis vectors
(REV-11); middleware error-path counting (REV-12); a Dagster schedule *tick* (REV-03); a mutated-
duplicate or null-valued-event parity case (REV-04/05); per-customer-`t0` parity in
`aggregate_stream` (all parity cases share one t0); an MLflow round-trip predict-equality assertion
(REV-13); a degenerate single-class bootstrap (REV-21). The pattern behind REV-02/03/12: the tests
assert *presence* (names, strings, schema) where they should assert *behavior* (a tick, a thrown
route, a lagged label).

---

## Architecture — portfolio assessment (senior-MLE lens)

**What is already staff-level (and under-marketed — say it out loud in README/WRITEUP):**
- Strict inward-only dependency direction: `src/churn` has zero imports from `api`/`services`/
  `orchestration` (verified). Most "MLOps portfolios" fail exactly this.
- Parity enforced as a **materializable Dagster gate** (`assets.py::parity_report` raises `Failure`
  at >1e-9), not just a test assertion.
- The `use_changelog_topics` checkpoint-ordering argument in `services/consumer/app.py` is real
  distributed-systems reasoning — promote it from docstring to WRITEUP.
- Typed seams in the right places (`ScoredCustomer.from_frame`, `LoadedModel`, `KVBackend`
  Protocol with Dict/Redis backends); registry provenance (git_sha + data_sha256 + lib versions).
- The promotion gate: paired ΔROC+ΔPR bootstrap on one shared draw matrix, MDE bar,
  incumbent-wins-ties. A worse-by-noise challenger cannot be promoted.

**The one architectural headline gap:** *promotion and serving are parallel universes.* The MLflow
`@champion` alias machinery (ADR 0004) cannot change what `serve.py`/`online.py` actually serve —
they load once from the file registry via `run_dir` and never invalidate. The lifecycle story ends
one step before production. Fixing this (serve-by-alias, a reload hook, or an explicit ADR saying
"cutover = redeploy, by design") is the single highest-signal improvement available.

**Ranked improvements (S/M/L effort · none touch the locked set):**
1. **Close the promotion→serving loop** (M) — serve resolves `@champion` (or file-registry
   `latest`) with an explicit reload/invalidation story, or an ADR documenting redeploy-as-cutover.
   Pairs with a **shadow scorer** (M/L): log challenger scores alongside champion on live traffic —
   the highest-signal net-new capability ("promotions validated on live traffic before cutover").
2. **Typed runtime Settings** (S) — `REDIS_URL`/`REDPANDA_BROKER`/`CHURN_*` env reads are scattered
   across 4+ files with duplicated defaults. One pydantic-settings class in the adapters layer +
   `.env.example`. Direct 12-factor signal; also subsumes ISS-11's config half.
3. **Feature-schema versioning on the online store** (M) — `feature_schema_version` + `as_of` in the
   Redis payload, checked on read (= the REV-11 fix). This is the "data contracts" interview topic.
4. **Alerts that fire** (S/M) — `rule_files` + p99-latency/5xx/scrape-down rules + a runbook stub.
   Converts "graphs" into "someone gets paged." (Depends on REV-12 to be truthful.)
5. **Real Dagster service in compose** (S) — `dagster dev -m orchestration.definitions` instead of
   the placeholder print. (Do REV-03 first so the schedule actually ticks.)
6. **Batch scoring as a scheduled asset** (M) — README declares batch "the production decision," but
   the recurring score job exists only as a CLI. A `batch_scores` asset on the weekly schedule
   closes the loop.
7. **Model card per run** (S) — `registry.save_run` metadata already holds 80% of one; render
   `MODEL_CARD.md` (intended use, honesty framing, limitations). Named artifact reviewers look for.
8. **AuthN or an explicit trust boundary** (S) — an API-key `Depends` on the routers, or one ADR
   line saying the boundary is the ingress. Interviewers ask.

**Conspicuous absences to be ready to discuss (fix or pre-empt in WRITEUP):** shadow/canary rollout
(item 1), data contracts (item 3), SLOs/alerting (item 4), CD beyond the manual placeholder, a
scalability/cost narrative (the O(1) aggregator genuinely scales — but Redis is single-node, DuckDB
single-process; write the 100x paragraph), model cards (item 7), API security (item 8).

---

## Suggested sequencing

1. **Quick wins, no design needed (do first):** REV-09, REV-12, REV-28, REV-15, REV-18, REV-23,
   REV-17's doc refreshes, REV-25. All small, all green-bar-safe.
2. **Grill-worthy (design decisions, then build):** REV-02 (horizon semantics), REV-03 (+item 5),
   REV-10 (degraded-contract shape), REV-11/item 3 (schema versioning), REV-14 (how the online app
   deploys), item 1 (promotion→serving), item 2 (Settings — coordinate with ISS-11).
3. **Post-re-lock backlog (ride along with ISS-15..18 whenever a re-lock is spent):** REV-01,
   REV-06, REV-07, REV-26, REV-27 — plus REV-08's option (a) if the CI-canonical confirmatory is
   ever re-run.
4. **Explicitly fine to defer:** REV-19, REV-20, REV-21, REV-22, REV-24, REV-29, REV-30.
