# Phase 0 — Lock & Pre-registration Decisions (v2.2, adds D5 frozen functional forms)

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
>
> **v2.2 (2026-07-01)** adds **D5 — the frozen simulator functional forms**, converged in an
> interactive grilling session at the start of the Phase-0 build (7 branches, each resolved with a
> recommendation the user accepted). D5 is the concrete specification that `docs/simulator_spec.md`,
> `simulator.params.json`, and the golden-vector tests implement. It changes nothing in D1–D4 — it
> *fills in* the "world" that D1–D4 freeze and guard.

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

---

## D5 — Frozen simulator functional forms (converged via grilling, 2026-07-01)

The concrete "world" that D1–D4 freeze and guard. Every equation and selector below is written into
`docs/simulator_spec.md` (prose) + `simulator.params.json` (numbers) and pinned by golden-vector
tests. Decided one branch at a time; each **Why** is the rationale that survived the grilling.

### Notation

`x` = a customer's static attributes; `φ(x)` = its encoding (one-hot categoricals + standardized
numerics); `h_w` = scalar latent health at week `w`; `W = 52`; `t0` = prediction time; `h(t0) = h_W`;
`σ(·)` = logistic sigmoid; `y` = churn label for the window `(t0, t0 + 90d]`.

### D5.1 — Latent health state (Q1)

**Scalar, continuous, weekly AR(1)/OU mean-reverting to a static-seeded baseline.**
```
h_0     = μ(x) + σ0 · z                      z    ~ N(0, 1)
h_{w+1} = h_w + κ · (μ(x) − h_w) + σ · ε_w    ε_w  ~ N(0, 1),   w = 0 … W−1
h(t0)   = h_W
```
**Why:** scalar keeps the oracle Bayes score 1-D and *analytically exact* (D4 demands "no fitted
estimator … the exact ranking ceiling"), makes golden vectors one-liners, and keeps the
common-cause story crisp. A low-dim vector latent is more realistic but makes the oracle a
multi-dim integral and complicates the freeze for no Phase-0 gain — noted as a possible Phase-4
realism extension.

### D5.2 — Hazard / label rule (Q2)

**A single Bernoulli draw at `t0`; logistic in `h(t0)` (with a frozen *negative* slope) and the
static quality index; static enters the hazard *directly*, not only through `h`.**
```
P(y = 1 | h(t0), x) = σ( α0 + α_h · h(t0) + α_stat · q(x) )    # frozen α_h, α_stat < 0
y ~ Bernoulli( P(y = 1 | h(t0), x) )
```
> **Sign convention (authoritative):** the additive form above with frozen **negative** `α_h`,
> `α_stat` is what `kernels.hazard_prob`, `simulator.params.json`, and `docs/simulator_spec.md` use.
> "Healthier / higher-quality → lower churn" comes from the negative coefficients, not from a minus
> sign in the formula. (Do not read this as `α0 − α_h·h` with positive magnitudes.)
**Why (evaluate at `t0`, not an integrated future-window hazard):** declaring `P(y | h(t0), x)`
*as* the logistic makes the oracle score **exactly** `σ(·)` with zero estimation and no future-window
RNG dragged into the ceiling. The label still *means* "churned in `(t0, t0+90d]`"; because `h` is
Markov the conditional is well-defined either way, and the single-logistic form is the one that is
analytically exact. **Why (static enters directly):** this is what D4 means by "the static
covariates that enter the frozen hazard" being oracle inputs — it gives static genuine predictive
value that is not merely a noisy proxy of `h`, matching the anchor where `subscription_plan` itself
carries signal. The oracle then legitimately needs both `h(t0)` and `q(x)`, and
`recoverable_lift = oracle_auc − static_auc` is exactly "the value of knowing `h(t0)` beyond static."

### D5.3 — Static/event column partition (Q3a)

**Partition the real columns by *nature*, not by the CSV layout.**

| Role | Real columns | Synthetic-world role |
|---|---|---|
| **Truly static** (seed `μ(x)` + hazard `q(x)`; anchor-fitted) | `region`, `device_type`, `subscription_plan`, `account_age_days`, `monthly_spend`, `avg_order_value` | customer attributes drawn from the anchor |
| **Event-derived** (summaries of the `h`-driven event stream) | `days_since_last_login`, `num_orders_last_90d`, `support_tickets_raised`, `pages_per_session` | **regenerated** from event streams in Phase 1 — the *event features* |

**Why:** in the real data the strongest numeric correlate, `days_since_last_login`, is conceptually
a *login-event summary*, not a fixed attribute. Putting it (and friends) on the event side is what
makes the positive control **meaningful rather than tautological**. Consequence, signed off:
`synthetic_static_auc` rests on customer attributes only and is **tuned** to ≈0.64 (the anchor
sanity band) via the direct static effect — legitimate frozen design, not fudging.

### D5.4 — Static → (health, hazard) map (Q3b)

**A single linear "customer-quality index" `q(x)`, entering both maps with two scales — not two
independent coefficient vectors.**
```
q(x)   = w · φ(x)          # w frozen; weighted so subscription_plan dominates (anchor-faithful)
μ(x)   = s_μ · q(x)        # health mean-reverts toward this
hazard = σ( α0 + α_h · h(t0) + α_stat · q(x) )    # frozen α_h, α_stat < 0 (see D5.2 sign note)
```
**Why:** parsimony + interpretability — one direction means "healthier-baseline" and
"directly-lower-churn" customers are the *same* people (realistic for churn), it halves the frozen
coefficient count, and golden vectors stay trivial (`q` is one dot product). Weighting `w` toward
`subscription_plan` reproduces the real "plan + a bit" structure and the monotone churn-by-plan
(Free .55 → Enterprise .32). `s_μ` and `α_stat` are the two knobs the scalar MC tunes.

### D5.5 — Event-emission link functions (Q4)

**Four families (per §4.2), each a canonical-link function of weekly health `h_w`. Coefficients set
a-priori (they do not affect the ceiling — see D5.9).**

| Family | Link | Direction | Emergent event feature(s) |
|---|---|---|---|
| **Login / engagement** | rate `λ_L(h) = exp(a_L + b_L · h)`; session depth `pages ~ N(a_P + b_P · h, τ_P²)` | healthier → more logins, deeper sessions | `days_since_last_login`, login freq/trend, `pages_per_session`, `num_orders_last_90d` |
| **Payment** | per-cycle failure `~ Bernoulli( σ(a_F + b_F · h) )` | healthier → fewer failures | payment-failure count |
| **Support** | rate `λ_S(h) = exp(a_S + b_S · h)`; sentiment `~ N(a_T + b_T · h, τ_T²)` | unhealthy → more tickets, more negative | `support_tickets_raised`, mean sentiment |
| **Downgrade** | per-cycle `~ Bernoulli( σ(a_D + b_D · h) )` | healthier → rarely downgrades | downgrade count / plan-change flag |

> **Sign convention:** every link is the additive form `a + b·h` (`kernels.exp_link` /
> `logistic_link` / `linear_link`). Frozen `b_L, b_P, b_O, b_T > 0` (healthier → more) and
> `b_F, b_S, b_D < 0` (unhealthy → more) — the direction lives in the sign of the frozen `b`.

**Why (pages + orders folded into the login/engagement family):** keeps to §4.2's four families;
session depth and purchase activity are engagement sub-signals of the same login process.
**Why (a-priori, generous login signal):** event coefficients govern *proxy quality*, i.e. how much
of the ceiling the pipeline can *recover* — not the ceiling itself. Setting baseline logins to
≈3–5/week (~150–250 over `W=52` → a strong `h`-proxy) with moderate slopes de-risks recovery.

### D5.6 — Time structure (Q5)

- **`W = 52` weeks (365d) pre-`t0`** feature window; `h` runs 52 weekly AR(1) steps to `h(t0)`.
  Sweet spot given `κ ≈ 0.2/wk` (mean-reversion washes out `h`-info older than ~1/κ ≈ 5–7 weeks, so
  beyond ~52wk adds little); consistent with the anchor's `days_since_last_login ∈ [0, 365]`.
- **Label window fixed `(t0, t0 + 90d]`** (unchanged from the spec).
- **Anchor experiment = one `(customer, t0)` row per anchor customer → n = 1,600** — exactly the `n`
  the MDE and confirmatory bootstrap are defined at.
- **The contract is panel-capable** (rows keyed by `(customer_id, t0)`; multiple `t0` per customer
  allowed in the ~50k population) — *why* §4.7's group-aware split by `customer_id` exists.

### D5.7 — Anchor / population construction (Q6)

- **~50k synthetic customers**; the **1,600 real are the distribution anchor.**
- **Fit to the real 1,600:** all **6 truly-static marginals** + the joints
  `subscription_plan × monthly_spend`, `plan × account_age_days`, `plan × region`; draw the 50k from
  the fitted joint (non-anchored pairs conditionally independent given plan).
- **Anchor rows keep the *real* static attrs**; their `h`-path, events, and label are **re-simulated**
  from the frozen world. The **real churn label feeds only `real_static_reference_auc`** — never the
  event experiment.
**Why:** makes the synthetic population production-shaped (R2's "distribution anchor, not a
real-outcome cohort"); re-simulating the anchor label keeps the anchor experiment honestly synthetic.

### D5.8 — Frozen control definitions (Q7)

Two complementary negative controls; both must show **no lift**:
- **Label-shuffle** — customer-level random permutation of `y`. A leak-free pipeline then scores
  **AUC ≈ 0.5, zero lift** for *any* feature set → catches label leakage anywhere.
- **Null-stream** — regenerate events with **all health-slopes zeroed**
  (`b_L=b_F=b_S=b_D=b_P=b_T=0` → events independent of `h`), keeping real static + labels. Static AUC
  is preserved but **event lift ≈ 0** → catches the pipeline hallucinating signal from noise /
  event-side leakage.
Control **definitions** are frozen in Phase 0; control **seeds** are free/exploratory during Phase-1
dev, and **beacon-derived (domain-separated tag)** if a control is reported in the confirmatory
tamper-evident artifact. **Why:** the pair brackets the failure modes — label-shuffle kills *all*
signal, null-stream preserves static but kills *event* signal — which the §4.7 sentinel suite then
covers path-by-path.

### D5.9 — Parameterization strategy & the recorded residual risk (freeze stance)

**Chosen stance: "tune-then-freeze (scalar MC)".** A ~50-line scalar Monte-Carlo — over
`(x → q(x) → μ(x), h(t0) ~ N, y ~ hazard)` with **no event generation** — solves for
`{α0, α_h, α_stat, s_μ, σ, σ0, κ}` to hit the D5-below targets, after which every number is frozen
into `simulator.params.json`.

**Ceiling tuning targets** (`oracle_auc`/`static_auc` depend only on latent+hazard+static, *not* on
events — the clean insight that keeps this harness event-free):

| Quantity | Target | Set via |
|---|---|---|
| Base churn rate | 0.46–0.50 | `α0` |
| `static_auc` (`q(x)` → churn) | 0.62–0.66 (anchor band) | `α_stat`, `μ→h` coupling |
| `oracle_auc` (true `h(t0)` + `q(x)`) | 0.80 ± 0.02 → `recoverable_lift ≈ 0.16` | `α_h`, variance split |
| `Var(μ(x)) : Var(stochastic h)` | **outcome** ≈ 1 : 13 (not an independent target) | falls out of the AUC targets |
| `κ` (mean-reversion) | 0.15–0.25 / week | — |

> **Disclosure — the variance split.** Q3c floated ≈1:1.3, but that is *incompatible* with a
> 0.64→0.80 recoverable gap: a large oracle-over-static headroom forces health to carry substantial
> information static cannot see, i.e. health must be **largely idiosyncratic**. The tuned world
> therefore lands at `Var(μ) : Var(stochastic h) ≈ 1 : 13.2` (`reports/simulator_tuning.json`; spec
> §3) — an *outcome* the AUC targets determine, not a free knob. Defensible and disclosed: static
> weakly seeds health while the **direct** static hazard term (`α_stat`) carries the plan-driven
> churn effect, and events (health proxies) reveal the idiosyncratic component — exactly why event
> features can lift AUC so much. `s_μ ≈ 0.28`, `α_h ≈ −1.24`, `α_stat ≈ −0.34` (`simulator.params.json`).

**Why this is not a peeking violation:** tuning the *world's* signal is legitimate experiment design
and is explicitly the "target-aware simulator tuning **guarded by the lock**" of §4.7 — the freeze,
before the analysis pipeline is built and before the confirmatory seed is drawn, is what makes it
honest. The confirmatory outcome (does the *pipeline* recover ≥50% of the headroom?) depends on the
Phase-1 analysis + a held-out beacon seed, which this tuning never touches. The floor is
`max(MDE, 0.5·recoverable_lift)`, computed at Stage 2 from the frozen world, so it **self-calibrates**
— the targets only need to keep the world non-degenerate.

**Recorded residual risk (accepted knowingly):** *recovery feasibility* — can the pipeline
reconstruct `h(t0)` from the a-priori event params well enough to clear the floor? — is **not tested
in Phase 0** (that needs event generation + feature aggregation + a model = Phase 1). Mitigation:
generous login signal (D5.5). If Phase-1 exploratory seeds show recovery is infeasible with the
frozen event params, that surfaces there and requires a **re-lock** (never a silent re-tune). The
alternative — pulling event-generation into Phase 0 to close this now — was considered and declined
to keep Phase 0 to the scalar-MC ceiling.

### D5.10 — Golden-vector coverage (what the conformance tests pin)

Deterministic fixtures pin every declared functional form against fixed `(inputs, params)`:
`q(x)`, `μ(x)`, the hazard `σ(α0 + α_h·h + α_stat·q)`, and each event link
(`λ_L`, `pages` mean, payment `σ`, `λ_S`, sentiment mean, downgrade `σ`). These catch a
"wrong-but-consistent" formula (e.g. a logistic link silently a step function) that the controls +
leakage sentinels would miss (D1 #1 code-fidelity). The stochastic *draws* (Exponential/Bernoulli/
Normal given a rate/prob/mean) are Phase-1 generation; the golden vectors pin the **links**, which
is the frozen math.

> **Refinement of D1 (world-construction code is hashed).** D1 originally left simulator *code*
> unhashed, relying on the golden vectors alone to pin behaviour. Two review rounds found the holes:
> a formula in `kernels.py` and its golden expected value could be **co-edited** with no drift; the
> `q(x)` encoder in `params.py` (imputation + standardization) could change health/hazard behaviour;
> and the **beacon round/KDF code** could be swapped (e.g. sha256→sha1) while the recipe *dict*
> `check_lock` compared stayed identical. So `HASHED_FILES` now hashes **`kernels.py`, `params.py`,
> and `beacon.py`** (all in the Layer-2 FROZEN regex, alongside `lock.py` and the workflow itself so
> a results PR can't weaken the guard), and `beacon.py` validates its randomness input. The golden
> vectors remain guard (2); hashing the code closes the co-edit / KDF-swap paths. The anchor
> `data/customer_data.csv` is pinned (raw hash) since the frozen standardization stats derive from
> it. The **floor design** (`n_oracle`, MDE replicate count + z-multiplier, `f=0.5`, metrics, floor
> entrypoint) is now recorded in the hashed `simulator.params.json` `floor_design`, so the spec's
> "Stage-1-fixed" claim is actually attested (Codex round). Every pass/fail RNG stream has a
> dedicated beacon domain tag (confirmatory + its bootstrap, oracle, MDE, floor-bootstrap, both
> controls).
