# Phase 0 — Lock & Pre-registration Decisions (v2.1, post 2nd Codex review)

> The Phase 0 lock decisions, to be folded into `docs/simulator_spec.md`, the analysis spec,
> `simulator.lock.json`, and the CI guard. **`ENHANCEMENT_PLAN.md` §4.6 now mirrors D1–D4
> authoritatively** (no longer a mere "working record" the plan ignores). **Not yet committed.**
>
> **v2 superseded v1** after a Codex adversarial soundness review (verdict: v1 not sound to
> freeze). The review's theme: v1 pinned the *world* tightly but left the *analysis side*
> (features, model, evaluation, the floor's statistical rigor) under-frozen — the main
> p-hacking surface. v2 closed that; all 11 findings folded in.
>
> **v2.1 folds in a 2nd Codex review's 3 findings:** **#1** plan-authority (ENHANCEMENT_PLAN.md
> §4.6 rewritten to make D1–D4 the executable plan); **#12** confirmatory-seed isolation (the seed
> is no longer committed in the clear at Stage 1 — see D3); **#13** a frozen oracle protocol so the
> oracle ceiling cannot move the pass/fail floor (see D4).

---

## Core principle (unchanged)

Freeze everything that determines the measured effect or the success criterion; freeze nothing
that is a free implementation choice. Enforce the freeze with a hash + a CI guard + behavioural
tests, before any confirmatory measurement. **Three distinct guards, never conflated:** the
**hash** proves *the declared rules didn't change*; **golden-vector tests** prove *the code
computes the declared functional forms*; the **controls + leakage-sentinel suite** prove
*the pipeline is leak-free and recovers signal*.

---

## The staged lock (the spine of v2)

- **Stage 1 — Phase 0:** freeze the **world**: latent process, event-emission functions,
  label/hazard rule, temporal contract, population + anchor, control definitions, the
  **evaluation protocol**, the **floor-derivation method** + the **frozen oracle protocol (D4)**,
  and the **confirmatory-seed commitment scheme** (the recipe, never the seed value — D3).
- **Stage 2 — end of exploratory Phase 1:** freeze the **analysis spec** (feature definitions,
  model + hyperparameters, preprocessing, metric/CI implementation, bootstrap unit/seed) and
  **compute + lock the numeric floors** by simulation on the now-frozen pipeline.
- **Then** the single **confirmatory run** on the held-out locked seed.

Both stages are locked *before* the confirmatory seed is ever touched, so the pre-registration
stays peek-free.

---

## D1 — What is frozen, and how (hash + golden tests + analysis-code hash)

**Hashed in the lock:**
1. `docs/simulator_spec.md` — prose: latent process, event-emission + hazard *functional
   forms*, label rule, control definitions, floor-derivation method.
2. `simulator.params.json` — every numeric constant + functional-form selectors + control
   definitions. **The confirmatory seed is NOT stored here in the clear** — only its
   commitment/derivation recipe is (see "Confirmatory-seed isolation" in D3).
3. **(Stage 2)** `analysis_spec.json` + an **analysis-code hash** over the feature/model/eval/
   measurement-entrypoint modules (the ANALYSIS set — see D2/#10). This is the deliberate
   refinement of v1's "never hash code": simulator *parameters* are hashed and its *behaviour*
   is pinned by golden tests; the **analysis pipeline is the experiment**, so its code is hashed
   and bound for confirmatory runs.

**Golden-vector conformance tests (closes #1 — code-fidelity).** Deterministic fixtures: fixed
inputs + params → expected event-emission and hazard outputs. These pin that the code computes
the *declared* functional forms (catching e.g. a logistic link silently implemented as a step
function — which controls + leakage sentinels would NOT catch). The fixtures are part of the
frozen spec set.

**Binding:** the simulator + analysis code read all numbers from the frozen JSON (single source
of truth, like `config.py`); they physically cannot use an unfrozen value. `simulator.lock.json`
records every hash, the **confirmatory-seed commitment/recipe (never the seed value before
Stage 2)**, git SHA, the (Stage-2) floors, and control defs — reusing `registry.py`'s `hashlib` +
`git_sha()`.

---

## D2 — CI guard (two layers, extended to the analysis code) + tamper-evident results

**Layer 1 — consistency.** A `lock` CI job recomputes every hash and asserts equality with
`simulator.lock.json`; non-zero exit on drift. Runs on push + PR. The measurement entrypoint
asserts the same at runtime and embeds the lock into results.

**Layer 2 — same-commit separation (extended, #10).** On PRs, `git diff --name-only
<merge-base>..HEAD` (checkout `fetch-depth: 0`); fail if the change set intersects **both**:
- FROZEN set = {`docs/simulator_spec.md`, `simulator.params.json`, `simulator.lock.json`,
  `analysis_spec.json`, **the ANALYSIS code modules**}, and
- RESULTS set = {`reports/instrument_validation/**`}.

This catches the v1 hole where feature/model code changed in the same commit as new passing
results.

**Governance (#2).** Protected branch; required review on any re-lock PR; no direct pushes that
bypass review. The procedural guard only works behind branch protection.

**Tamper-evident results (#3; hardened for re-review #2).** The committed results JSON carries
provenance: lock hash, analysis-code hash, git SHA, clean-tree marker, the seeds. CI does **not**
trust the committed predictions: it **regenerates the raw confirmatory predictions from scratch**
in a clean checkout from the locked spec/params + the beacon-derived seed + the hashed analysis
code (canonical row ordering + serialization; bit-reproducible given the pinned dependency/threading
versions), compares them **row-by-row by content hash** to the committed predictions, and **only
then** recomputes the summary (ΔROC/ΔPR-AUC CIs, verdict) and asserts agreement. So neither a
hand-edited summary **nor a fabricated-but-self-consistent prediction artifact** can pass — the
predictions must reproduce the frozen pipeline. CI **likewise regenerates the locked numeric floors**
from `R` + the frozen pipeline and asserts they equal the committed floors, so the **bar** cannot be
hand-set either. Heavy artifacts (plots) stay gitignored; raw predictions + summary are small and
committed.

**Limitation (R3, unchanged).** Procedural, not cryptographic, un-gameability; mitigated by the
conspicuous, reviewed, results-free re-lock commit behind branch protection.

---

## D3 — Spec depth + two-seed discipline + the analysis freeze (staged)

**Stage 1 frozen (the world + criterion + protocol):** latent process (form + params +
static-attribute→seed mapping + the invariant *"label is never an input"*); event-emission
functions (forms + params); label/hazard rule + `(t0, t0+90d]` window + `features ∈ (−∞, t0]`;
population (size, anchor role, which marginals/joints fit the real 1,600, anchor-label
re-simulation); control definitions; the evaluation protocol; the floor-derivation method;
**the frozen oracle protocol (D4)**; **the confirmatory-seed commitment scheme (not the value)**.

**Stage 2 frozen — the analysis spec (closes #4, #5):** exact **feature definitions** (eligible
aggregations + windows + inclusion rules); **model family + hyperparameters** (inherits the base
project's tie-aware LogReg); preprocessing; the **selection metric**; the **bootstrap unit =
customer** (not row — respects the group split) + method; the CI method — **the bootstrap *seed* is
beacon-`R`-derived per D4, not chosen here**. Frozen on exploratory seeds, before any confirmatory draw.

**Two-seed split:** exploratory seed(s) (unlocked — develop + look freely) vs the
**confirmatory seed** (single shot). The held-out seed alone does NOT close feature p-hacking
(exploratory and confirmatory share the frozen world); the **analysis freeze** is what closes it.

**Confirmatory-seed isolation (closes #12; hardened for re-review #1).** The confirmatory seed is
**never committed in the clear at Stage 1** — otherwise a local pre-Stage-2 simulator run against
the known seed could silently guide feature/model/floor choices, leaving no committed trace (the
"peek then tune" hole). The recipe is **fully deterministic** — a postdating constraint *alone* is
insufficient, because it would still let a solo author shop among eligible beacon rounds. Stage 1
locks **all four** of:
- **Beacon source** — the drand League-of-Entropy chain; its **chain hash + genesis + period** are
  recorded in `simulator.lock.json` (so "the beacon" itself cannot be swapped).
- **KDF + canonicalization** — `seed_entropy = SHA256("churn-confirmatory" ‖ beacon_randomness(R))`,
  fed to a NumPy `SeedSequence` (exact byte order pinned).
- **Exact round rule** — `R = the first chain round whose beacon time ≥ (T_trusted + Δ)`, where
  `Δ` is fixed at Stage 1 and **`T_trusted` is a server-controlled timestamp, never the author's
  git commit date**: the protected-branch merge / CI first-seen time of the Stage-2 lock, recorded
  by CI. drand rounds are a deterministic function of time, so `R` is **uniquely determined** by
  `T_trusted` — exactly one eligible round, whose value does not yet exist (see CI enforcement).
- **CI enforcement (closes re-review #1, the timestamp hole).** CI: (a) reads `T_trusted` from the
  protected-branch merge/CI event, **not** from `git`; (b) **rejects** the Stage-2 lock if the
  commit author/committer date is outside a small skew window of `T_trusted` (no backdating to land
  an already-emitted round); (c) requires round `R` to be **unemitted at first verification**
  (`T_trusted + Δ` strictly in the future), so its value cannot already be known; (d) recomputes
  `R`, re-derives the seed, and **rejects any other `R` or seed**; (e) records `T_trusted` + `R` in
  the results. The trusted clock is not the author's, so a solo author cannot land a preferred round
  by choosing the commit time.

The value is unknowable — to anyone, **including a solo author** — until round `R` emits, strictly
after the analysis is frozen. A **sealed hash** (`sha256(seed‖nonce)`, preimage held by a non-author
reviewer) is reserved for **genuine multi-party custody only**; with a self-held preimage it binds
only against *changing* the seed, not self-peeking, so it is **not** used here.

`simulator.lock.json` records the **beacon parameters + round rule** (never a pre-Stage-2 seed
value); the emitted seed is recorded only in the (tamper-evident) confirmatory results.

**Stopping rule (strict).** A confirmatory miss is a **recorded null** for that lock. No silent
bug-fix-and-rerun: any re-run requires a **reviewed re-lock**, which produces a **fresh Stage-2
commit** and therefore — via the deterministic round rule above — a **fresh, still-unpredictable
beacon-derived seed** (never a hand-picked or "sealed" seed). You can never quietly roll the dice
twice, and you cannot choose the new dice.

---

## D4 — The floor = two gates: `max(MDE, f · recoverable_lift)`

Per metric (ROC-AUC **and** PR-AUC); the confirmatory paired-bootstrap CI lower bound must
exceed the floor for **both**.

**Floor RNG is held out exactly like the confirmatory seed (closes re-review #3, round 3).** The
floor-computation *design* — `N_oracle`, MDE replicate counts, the domain tags, the floor
entrypoint, dependency versions — is fixed at Stage 1; but **every floor RNG stream is derived from
the same trusted-timestamp beacon round `R` as the confirmatory seed, via domain-separated tags**
(`KDF(beacon_randomness(R), "oracle" | "mde" | "floor-bootstrap")`). So the floor draws are (a) **not
author-chosen** — the beacon is unpredictable (round-2 #3) — and (b) **not knowable during
exploratory Phase 1** — `R` postdates the analysis lock (round-3 #3); you cannot shop the analysis
spec against known floor draws. Domain-separated KDF streams from `R` are statistically independent,
and the floor's large-sample draws share **no observations** with the `n=1,600` confirmatory draw,
so reusing `R` introduces no dependence between the bar and the measurement. After `R` emits the
floor is a deterministic function of (frozen pipeline + `R`), recorded **before** the verdict is
taken; CI **regenerates it** and asserts it equals the committed floor, exactly as it regenerates
the confirmatory predictions (D2).

**Gate 1 — detectability (proper MDE, closes #6/#7).** Predeclared paired Monte-Carlo
simulation at the anchor `n = 1,600` → empirical `SE_ΔAUC` (the *paired* run captures score
covariance), with the **MC replicate count fixed at Stage 1 and the MC seed(s) derived from beacon
round `R`** (domain tag `"mde"`).
`MDE = (z_{1−α} + z_{power}) · SE_ΔAUC`; for one-sided α=0.05 and **power = 0.80**, ≈ `2.49 · SE`.
Computed separately for ROC-AUC and PR-AUC (PR-AUC via `average_precision`).

**Gate 2 — substantive recovery (closes #8 + #13).** An **oracle ceiling**: the best any function
of events could achieve, given the common-cause structure (events ⟂ label | latent state). The
oracle protocol is **frozen at Stage 1 with the same specificity as the analysis pipeline**, so it
cannot move the floor after the fact:

- **Inputs:** the **true latent health state at `t0`** (`h(t0)`) plus exactly the static
  covariates that enter the frozen hazard — and **nothing post-`t0`**.
- **Score:** the **analytic Bayes-optimal score** — the simulator's *true* conditional churn
  probability for `(t0, t0+90d]` under the frozen hazard, `P(y=1 | h(t0), x_static)`. There is
  **no fitted estimator**, hence no model/hyperparameter/training-seed freedom; this is the exact
  ranking ceiling, not an approximation.
- **Evaluation:** on `N_oracle = 200_000` independent customers drawn from the frozen population
  using **`oracle_seed` derived from beacon round `R`** (`KDF(beacon_randomness(R), "oracle")` —
  neither chosen at Stage 2 nor knowable before the analysis lock), with `static_AUC` computed by
  the *frozen analysis model* on the same draw; ROC-AUC via `roc_auc_score`, PR-AUC via
  `average_precision`.
- **Uncertainty + rounding:** report the bootstrap SE/CI of `oracle_AUC`, `static_AUC`, and
  `recoverable_lift`; carry **full precision** into the floor (no rounding — the floor never
  depends on a rounded reference, per #9).

`recoverable_lift = oracle_AUC − static_AUC`. Substantive floor = **`f = 0.5 · recoverable_lift`**.
**`f = 0.5` is pre-registered, not tuned** — recovering *half* the achievable headroom is the
smallest fraction that reads as "materially recovers the signal" rather than merely "detectably
non-zero" (Gate 1 already owns detectability). Claim becomes concrete: *"recovers ≥ 50% of the
signal an oracle with perfect latent-state knowledge could extract."*

**Success:** confirmatory paired ΔAUC CI lower bound > `max(MDE, 0.5 · recoverable_lift)`, both
metrics. **Reported alongside** (not gated): the *actual* recovered fraction, the *actual* power
at the final floor (if `n=1,600` can't reach 80%, disclosed as a scale limitation), the oracle
ceiling, and the exact reference AUC + uncertainty (#9 — the restructure no longer depends on
the rounded ≈0.64; the real reference is reported alone per §4.4).

**Pinned implementation (#11):** percentile bootstrap, **customer-level** resampling,
`average_precision` for PR-AUC, the oracle protocol's **inputs/analytic score/`N_oracle`**, MDE
**replicate counts + domain tags + the floor entrypoint**, and key dependency versions are fixed at
**Stage 1** (in `analysis_spec.json` / the lock); but **all pass/fail-determining RNG *seeds* —
`oracle_seed`, the MDE Monte-Carlo seed(s), and the confirmatory paired-bootstrap seed — are derived
from the beacon round `R`** via domain-separated tags (never stored as free values), so they are
neither author-chosen nor knowable before the analysis is frozen.
