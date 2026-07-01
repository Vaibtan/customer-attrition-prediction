# Simulator Specification (frozen, Stage-1 pre-registration)

> **Status:** Stage-1 frozen. This document + `simulator.params.json` + the world-construction code
> (`src/churn/simulator/kernels.py`, `params.py`, `beacon.py`) are hashed into `simulator.lock.json`
> (and the anchor `data/customer_data.csv` is pinned by a raw hash) and guarded by CI (Layer-1
> consistency + Layer-2 same-commit separation) *before* any confirmatory measurement. This is the
> **prose** (functional forms, protocol, definitions); every **number** lives in
> `simulator.params.json` (single source of truth). The
> decision record + rationale is `PHASE0_LOCK_DECISIONS.md` (D1–D5); this file is the operational
> frozen artifact the code and the measurement entrypoint read.
>
> **Frozen here (Stage 1):** the world (latent process, event/hazard forms, label rule, temporal
> contract, population + anchor), the control definitions, the evaluation protocol, the **oracle /
> floor-derivation _method_**, and the **confirmatory-seed + floor-RNG _commitment scheme_**.
> **NOT here (Stage 2, end of Phase 1):** the analysis spec, the analysis-code hash, and the
> **numeric floor values** (computed from the frozen pipeline after the beacon round `R` emits).

---

## 0. What "frozen" buys, and the three guards

We freeze everything that determines the measured effect or the success criterion, and nothing
that is a free implementation choice. Three **distinct** guards enforce it, never conflated:

1. **Hash** (`simulator.lock.json`) **attests** *the declared rules did not change*.
2. **Golden-vector conformance tests** (`tests/test_simulator_kernels.py`,
   `tests/test_simulator_params.py`) **pin** *that the code computes the declared functional forms* —
   catching a wrong-but-consistent formula (e.g. a logistic link silently a step function) that the
   controls would miss.
3. **Controls + the leakage-sentinel suite** (Phase 1) **argue** *leak-freeness (never proved) and
   check that the pipeline recovers signal* — leak-freeness is argued from the suite plus the
   negative controls, never claimed as proof.

The scoped claim this instrument supports is **synthetic**: "it recovers injected signal (positive
control), reports nothing when there is nothing (negative controls), and passes a leakage-sentinel
suite." **No real-world AUC claim is made without real event data.**

---

## 1. Population & anchor (D5.3, D5.7)

- **~50,000 synthetic customers** (`population.size`). The **1,600 real customers**
  (`data/customer_data.csv`) are a **distribution anchor**, not a real-outcome cohort.
- **Fit to the anchor:** the 6 truly-static marginals (`population.fit_marginals`) and the joints
  `population.fit_joints` (`subscription_plan × monthly_spend`, `× account_age_days`, `× region`).
  Non-anchored pairs are drawn conditionally independent given `subscription_plan`.
- **Static vs event-derived columns (partition by nature, not CSV layout):**
  - **Truly static** (customer attributes; seed health `μ(x)` + enter the hazard):
    `region`, `device_type`, `subscription_plan`, `account_age_days`, `monthly_spend`,
    `avg_order_value`.
  - **Event-derived** (regenerated from the health-driven event stream; these are the *event
    features*): `days_since_last_login`, `num_orders_last_90d`, `support_tickets_raised`,
    `pages_per_session`.
- **Anchor experiment:** the anchor rows keep their **real static attributes**; their health path,
  events, and **label are re-simulated** from this frozen world. The **real churn label feeds only
  `real_static_reference_auc`** (§6) — never the event experiment.

---

## 2. Customer-quality index `q(x)` (D5.4)

A single standardized index over the static attributes (`plan` dominates; `region`/`device_type`
carry zero weight, matching the anchor where they add ~nothing):

```
q_raw(x) = w_plan·plan_score(x) + w_age·z(account_age_days)
         + w_spend·z(monthly_spend) + w_aov·z(avg_order_value)
q(x)     = ( q_raw(x) − mean_q_raw ) / std_q_raw           # unit variance over the anchor
```

- `plan_score`: `quality_index.plan_score_map` (Free 0 → Enterprise 3; monotone → healthier).
- `z(·)`: standardize with the frozen per-numeric median (impute) + mean/std in
  `quality_index.numeric_standardize`.
- weights `w_*`: `quality_index.weights`; `mean_q_raw`, `std_q_raw`: `quality_index.q_raw_standardize`.

Implemented once in `churn.simulator.params.quality_index`; kernel: `kernels.quality_index_raw`.

---

## 3. Latent health state `h` (D5.1)

Scalar, continuous, mean-reverting **weekly AR(1)/OU** toward a static-seeded baseline, over a
`latent.weeks` (= 52) window:

```
μ(x)     = s_μ · q(x)                                   # kernels.latent_baseline
h_0      = μ(x) + σ0 · z ,           z   ~ N(0,1)
h_{w+1}  = h_w + κ·(μ(x) − h_w) + σ·ε_w ,  ε_w ~ N(0,1),  w = 0 … 51    # kernels.latent_step
h(t0)    = h_52
```

Numbers: `s_μ = quality_index.s_mu`; `κ = latent.kappa`, `σ = latent.sigma`, `σ0 = latent.sigma0`.
At `κ = 0.05` (D7 re-lock; ~20-week memory) the process is stationary well within 52 weeks, so
`h(t0) ~ N(μ(x), σ0²)`. `κ` is **ceiling-neutral** — the stationary `h(t0)` distribution the oracle
and static AUCs depend on does not involve `κ` — so it governs only how *observable* the frozen
health signal is from the event history (a proxy-quality knob; see §7 provenance and D7).

**Health is weakly static-seeded and largely idiosyncratic** (frozen split `Var(μ) : Var(stochastic
h) ≈ 1 : 13`). This is the *intended, disclosed* consequence of a large recoverable gap (§7): if
events are to lift AUC from ≈0.64 to ≈0.80, health must carry substantial information that static
attributes cannot see. The direct static hazard term (§4) captures the plan-driven churn effect that
is separate from engagement-health.

---

## 4. Hazard / label rule + temporal contract (D5.2, D5.6)

**Per row `(customer_id, t0)`:** predict at `t0`; features from events in `(−∞, t0]`; the label is
**churn in `(t0, t0 + 90d]`** (`hazard.label_window_days` = 90). The point-in-time contract
`max(feature_ts) ≤ t0 < min(label_ts)` is enforced by the Phase-1 PIT sentinel. The contract is
**panel-capable** (`temporal.row_key = [customer_id, t0]`); the n = 1,600 anchor confirmatory set
uses **one row per customer** (`temporal.anchor_rows_per_customer` = 1).

The label is a **single Bernoulli draw at `t0`**, logistic in health and quality:

```
P(y = 1 | h(t0), q(x)) = sigmoid( α0 + α_h·h(t0) + α_stat·q(x) )        # kernels.hazard_prob
y ~ Bernoulli( P(y = 1 | h(t0), q(x)) )
```

Numbers: `hazard.alpha0`, `hazard.alpha_h`, `hazard.alpha_stat` (both slopes frozen **negative**:
healthier / higher-quality → lower churn). Evaluating the hazard **at `h(t0)`** (not integrating a
future-window hazard) makes this expression *exactly* the analytic Bayes-optimal oracle score (§8) —
no fitted estimator, no future-window RNG. **The churn label is never an input to event generation.**

---

## 5. Event-emission functional forms (D5.5)

Four families, each a **canonical-link function of the weekly health `h_w`**. The links are frozen
here; the stochastic *draws* (Exponential/Bernoulli/Normal given a rate/prob/mean) are Phase-1
generation. Event coefficients are set **a-priori** (they govern proxy quality — how much of the
ceiling the pipeline can *recover* — not the ceiling itself), so they are **not** MC-tuned.

| Family | Link (kernel) | Numbers | Emergent event feature(s) |
|---|---|---|---|
| Login / engagement | rate `λ_L(h)=exp(a_L+b_L·h)` (`exp_link`); depth `pages~N(a_P+b_P·h, τ_P²)` (`linear_link`); orders `λ_O(h)=exp(a_O+b_O·h)` | `events.login_rate`, `events.session_depth`, `events.order_rate` | `days_since_last_login`, login freq/trend, `pages_per_session`, `num_orders_last_90d` |
| Payment | failure `~Bernoulli(sigmoid(a_F+b_F·h))` (`logistic_link`) | `events.payment_fail` | payment-failure count |
| Support | rate `λ_S(h)=exp(a_S+b_S·h)` (`exp_link`); sentiment `~N(a_T+b_T·h, τ_T²)` (`linear_link`) | `events.support_rate`, `events.sentiment` | `support_tickets_raised`, mean sentiment |
| Downgrade | `~Bernoulli(sigmoid(a_D+b_D·h))` (`logistic_link`) | `events.downgrade` | downgrade count / plan-change flag |

**Sign convention (frozen in the numbers):** `b_L, b_P, b_O, b_T > 0` (healthier → more logins,
deeper sessions, more orders, more positive sentiment); `b_F, b_S, b_D < 0` (unhealthier → more
payment failures, more support tickets, more downgrades). Baselines at `h = 0` are the designed
semantics (D7 re-lock: ~6 logins/wk, ~8 pages/session, ~1 order/wk, ~7% payment failure,
~0.4 tickets/wk, neutral sentiment, ~3% downgrade/cycle) — pinned by the binding golden vectors.
The event coefficients set **proxy quality** (how strongly and cleanly events reveal `h`), not the
ceiling (§7/§8); they were strengthened in the D7 re-lock to make recovery feasible.

> **Frozen vs Phase-1 boundary (scope, disclosed).** Stage 1 freezes the event **links** (rate /
> prob / mean as a function of `h`) + the declared **draw families** (Exponential / Bernoulli /
> Normal) + all coefficients. It does **not** freeze the generator *implementation* — per-week event
> timestamp placement, sampling cadence, any clipping, and the population sampler are **Phase-1
> code**, pinned there by the golden vectors, the leakage-sentinel suite, and the Stage-2 analysis
> freeze. Because event coefficients don't affect the ceiling (§7/§8), this is the "tune-then-freeze
> (scalar MC)" scope decision (D5.9), and it leaves the **recovery-feasibility residual risk** stated
> in D5.9: whether the Phase-1 pipeline can actually reconstruct `h(t0)` from these events is not
> tested until Phase 1, and an infeasible-recovery finding forces a re-lock, never a silent re-tune.

---

## 6. Evaluation protocol — three named baselines (D5, §4.4 of the plan)

Reported under **fixed names, never substituted** for one another:

- **`real_static_reference_auc`** — static features, **real** labels, anchor (≈0.64). Reported
  **alone**; never compared against event-augmented AUC (apples-to-oranges: the event experiment
  uses *simulated* labels).
- **`synthetic_static_auc`** — static features, **simulated** labels, anchor. Sanity: lands near the
  anchor band (≈0.64) — the frozen world is *tuned* to this (§7), not inheriting it.
- **`synthetic_static_plus_event_auc`** — static **+ event** features, **simulated** labels, anchor.
  The positive-control headline.

**Positive control:** `synthetic_static_plus_event_auc − synthetic_static_auc` clears the Stage-2
locked floor on a paired-bootstrap **ΔROC-AUC and ΔPR-AUC** CI lower bound. **Negative controls**
(§9) show **no lift**. The leakage-sentinel suite (Phase 1) must pass.

---

## 7. Frozen world regime (provenance)

Set by `churn.simulator.tuning` (scalar Monte-Carlo over `q → μ → h(t0) → y`, **no event
generation** — the ceiling depends only on latent + hazard + static). Achieved regime recorded in
`reports/simulator_tuning.json`; targets in `tuning.targets`:

| Quantity | Target | Achieved (dev seed, N=300k) |
|---|---|---|
| base churn rate | 0.48 | ≈0.478 |
| `synthetic_static_auc` | 0.64 | ≈0.639 |
| `oracle_auc` | 0.80 | ≈0.803 |
| `recoverable_lift = oracle − static` | — | ≈0.164 |

Tuning the world's signal, then **freezing** it, is the legitimate "target-aware simulator tuning
guarded by the lock" (plan §4.7). It never touches the Phase-1 analysis pipeline or the confirmatory
seed, so pre-registration integrity holds. The non-degeneracy check
(`tests/test_simulator_params.py::test_frozen_world_has_real_recoverable_headroom`) re-derives the
regime from the frozen params via an independent MC.

---

## 8. Frozen oracle protocol (D4, method only)

An **oracle ceiling** = the best any function of events could achieve given the common-cause
structure (events ⟂ label | latent state). Frozen at Stage 1 with the same specificity as the
analysis pipeline, so it cannot move the floor after the fact:

- **Inputs:** the **true latent health `h(t0)`** plus the static covariates that enter the hazard
  (here summarized by `q(x)`) — and **nothing post-`t0`**.
- **Score:** the **analytic Bayes-optimal** `P(y=1 | h(t0), q(x))` = the frozen hazard (§4). **No
  fitted estimator** → no model/hyperparameter/training-seed freedom; the exact ranking ceiling.
- **Evaluation:** on `N_oracle` independent customers from the frozen population, with `static_auc`
  from the frozen analysis model on the same draw; ROC-AUC and PR-AUC (`average_precision`).
- **`recoverable_lift = oracle_auc − static_auc`**, carried at **full precision**.

`N_oracle`, the MDE replicate counts, the domain tags, and the floor entrypoint (the floor
**design**) are fixed at Stage 1 and **recorded concretely in the hashed `simulator.params.json`
`floor_design`** — `n_oracle = 200,000`, `mde_paired_mc_replicates = 2000`, `mde_z_multiplier = 2.49`
(one-sided α=0.05, power=0.80), `substantive_fraction_f = 0.5`, `metrics = [roc_auc, pr_auc]`,
`floor_entrypoint = churn.instrument.floors:compute_floors`, `oracle_score = analytic_bayes_hazard`.
Only the floor **RNG seeds** and the numeric floor **values** are held out (§10, Stage 2).

## 8b. Frozen floor-derivation method (D4, method only — numeric floors are Stage 2)

Per metric (ROC-AUC **and** PR-AUC), the confirmatory CI lower bound must exceed:

```
floor = max( MDE , 0.5 · recoverable_lift )
```

- **Gate 1 — MDE (detectability):** predeclared **paired** Monte-Carlo at `n = 1,600` → empirical
  `SE_ΔAUC`; `MDE = (z_{1−α} + z_power)·SE_ΔAUC`; for one-sided α=0.05, power=0.80, ≈ `2.49·SE`.
- **Gate 2 — substantive recovery:** `0.5 · recoverable_lift` — recovering **half** the achievable
  headroom (pre-registered `f = 0.5`, not tuned). Claim: "recovers ≥ 50% of the signal an oracle with
  perfect latent-state knowledge could extract."

**The numeric floor values are NOT computed here.** They are computed at Stage 2 from the frozen
pipeline after beacon round `R` emits, and CI re-derives them from `R` (D4).

---

## 9. Control definitions (frozen; D5.8)

Two complementary negative controls, both of which must show **no lift**:

- **Label-shuffle** (`controls.label_shuffle`) — customer-level random permutation of `y`. A
  leak-free pipeline then scores **AUC ≈ 0.5, zero lift** for *any* feature set → catches label
  leakage anywhere.
- **Null-stream** (`controls.null_stream`) — regenerate events with **all health-slopes `b`
  zeroed** (events independent of `h`), keeping real static + labels. Static AUC is preserved but
  **event lift ≈ 0** → catches the pipeline hallucinating signal from noise / event-side leakage.

Control **definitions** are frozen here; control **seeds** are free/exploratory during Phase-1 dev,
and **beacon-derived (domain-separated tag)** if a control is reported in the confirmatory
tamper-evident artifact.

---

## 10. Confirmatory-seed & floor-RNG isolation (D3/D4, commitment scheme only)

**No seed is committed in the clear.** The confirmatory seed **and** all pass/fail-determining floor
RNG (`oracle_seed`, MDE Monte-Carlo seed(s), floor/confirmatory bootstrap seeds) derive from a
single pinned **drand public-randomness beacon round `R`** via domain-separated KDF tags. The pinned
chain parameters (chain hash + genesis + period), the KDF, the exact round rule
(`R = first round at time ≥ T_trusted + Δ`, `T_trusted` = the protected-branch merge / CI
first-seen time, **never the author's git commit date**), and the CI enforcement live in
`simulator.lock.json` and `PHASE0_LOCK_DECISIONS.md` D3/D4. The `dev_seed` in
`simulator.params.json` (`tuning.dev_seed`) is a **development** seed for the tuning MC **only** — it
is explicitly **not** the confirmatory seed.

---

## 11. What is deferred to Stage 2 (end of Phase 1)

Frozen there, **not** here: `analysis_spec.json` (feature definitions, model family +
hyperparameters, preprocessing, selection metric, bootstrap unit = customer + method, CI method),
the **analysis-code hash**, and — after `R` emits — the **numeric floor values** computed from the
frozen pipeline. See `IMPLEMENTATION_CHECKLIST.md` (end of Phase 1).
