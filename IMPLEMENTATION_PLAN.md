# Implementation Plan — Customer Churn Prediction

> **Status:** Approved scope (decisions locked via interview, see §3).
> **Author:** ML Engineering
> **Audience:** This document is written to be read by a senior MLE / CTO. It states a thesis, defends the non-obvious calls with evidence from the data, and maps every line of work back to either the exercise spec or a production concern.

---

## 1. Executive summary (the thesis)

I profiled the 1,600-row dataset before writing a line of modelling code. Two findings drive every decision in this plan:

1. **The label is essentially balanced — 48.4% churn.** The README's phrase *"class imbalance"* refers to the **`subscription_plan`** distribution (Free 733 → Enterprise 117), **not** the target. Reflexively reaching for SMOTE / oversampling / `class_weight='balanced'` here would be a tell of a junior analyst and would *degrade* calibration. We will explicitly **not** resample, and will say why.

2. **The predictive signal is genuinely weak — and that is the interesting part.** A *leak-free* CV tops out at **ROC-AUC ≈ 0.59–0.64**. Just two fields — `subscription_plan` + `days_since_last_login` — carry almost all of it; the other nine add little. Crucially, the three models are a **statistical tie** (CV ROC-AUC: LogReg **0.594 ± 0.027**, Random Forest 0.592, HistGB 0.565; the paired hold-out AUC-gap bootstrap CI **straddles 0**), so we select **Logistic Regression for interpretability + calibration, not because it "wins"**. Model complexity buys nothing on this data.

| Model (RepeatedStratifiedKFold 5×3, leak-free pipeline) | CV ROC-AUC | Hold-out F1 @0.5 |
|---|---|---|
| Logistic Regression | **0.594 ± 0.027** | 0.583 |
| Random Forest (300 trees) | 0.592 ± 0.027 | 0.559 |
| HistGradientBoosting | 0.565 ± 0.025 | 0.531 |

The top two are a statistical tie (§4); the paired hold-out AUC-gap bootstrap CI
straddles 0, so LogReg is *selected*, not crowned. A feature ablation (same
leak-free CV) shows why everything else barely matters — and, tellingly, that the
two key fields **alone beat the full feature set**, i.e. the other nine add noise:

| Feature subset (LogReg, CV ROC-AUC) | ROC-AUC |
|---|---|
| `subscription_plan` only | 0.578 |
| `days_since_last_login` only | 0.587 |
| `subscription_plan` + `days_since_last_login` | **0.619** |
| All features (full pipeline) | 0.594 |

**Strategic consequence.** A portfolio submission to a senior reviewer lives or dies on whether it *honestly diagnoses* this ceiling and then **manufactures business value from a modest model** (calibrated probabilities + cost-based thresholding + uplift framing), rather than fabricating a suspicious 0.95-AUC. That honest-diagnosis-plus-value narrative is the chosen story (§3, decision D3).

**The single most important starter-code defect is not one of the three labelled bugs — it is data leakage:** outlier removal, imputation, and scaling are all fit on the *entire* dataset before the train/test split. Everything we build is organized around a leak-free `Pipeline` + `ColumnTransformer` so that all fitted statistics are learned per training fold only.

---

## 2. Data audit (evidence behind the thesis)

Profiled with the actual CSV — all numbers below are measured, not assumed.

### 2.1 Shape & integrity
- **1,600 rows × 12 columns.** No duplicate `customer_id`, no duplicate rows.
- Target `churned`: mean **0.4838** → balanced.

### 2.2 Missing values — **234 rows (14.6%) affected**, 80 cells (5.0%) per column
| Column | Missing | Pattern |
|---|---|---|
| `monthly_spend` | 80 (5%) | scattered |
| `avg_order_value` | 80 (5%) | scattered |
| `pages_per_session` | 80 (5%) | scattered |

No row is missing all three (the three missing-sets don't co-occur). Missingness is plausibly **MAR**; we will add **missingness indicators** (`add_indicator=True`) so the model can exploit any informative-missingness signal rather than silently erasing it.

### 2.3 Impossible / inconsistent values — **29 distinct rows (1.8%)**
| Column | Valid range | Issue | # rows |
|---|---|---|---|
| `account_age_days` | `[0, ~1500]` (observed max 1496) | negative (`-1`, …) | 8 |
| `days_since_last_login` | `[0, 365]` | negative | 3 |
| `days_since_last_login` | `[0, 365]` | sentinel values `800`, `999` | 7 |
| `num_orders_last_90d` vs spend | ~~orders = 0 ⇒ spend = 0~~ **(rejected, see §5)** | `orders == 0` with `spend > 0` is **valid** (90-day vs 6-month window) | 11 |

### 2.4 Outliers — the "small proportion of outlier spend values"
| Column | Median | p99 | Max | # > 1000 |
|---|---|---|---|---|
| `monthly_spend` | 56 | 10,318 | **18,772** | 21 |
| `avg_order_value` | 122 | 5,423 | **9,724** | 24 |

These are 1–2 order-of-magnitude spikes — leverage points that wreck a scaled linear model. We **winsorize (cap), not delete** (§5, rationale in D-clean).

### 2.5 Signal map (what actually predicts churn)
- **Point-biserial correlations with `churned`** (clean, imputed): everything is weak — `days_since_last_login` **+0.151** is the strongest; all others |r| ≤ 0.07.
- **Churn rate by plan** (real categorical signal): Free **0.551** → Basic 0.472 → Premium 0.396 → Enterprise **0.316**. Monotone with plan "tier."
- **Churn rate by region** (0.448–0.509) and **by device** (Tablet 0.428 → Desktop 0.503): negligible.
- **Median feature by class:** churned customers log in less recently (199 vs 157 days) — and that's essentially the whole story.

> **Read:** dormant, low-tier (Free) customers churn. The model is a recency-and-plan detector with a thin tail of additional signal.

---

## 3. Decision log (resolved in interview)

These are mini-ADRs. Each was an explicit fork; the chosen option is recorded with its rationale.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| **D1** | Deliverable shape | **Full MLOps showcase** — `src/` package + tests + FastAPI demo + local registry + monitoring + CI + Docker, fronted by a runnable `churn_prediction.py` | Maximises engineering signal. *Risk:* over-engineering optics for an 1,800-row CSV → mitigated by making every component demonstrate **domain judgment** (D2) and keeping infra lightweight where heavy infra adds no insight (D4). |
| **D2** | Serving design | **Batch-first; REST as labelled demo** | Churn scoring for retention is a **batch** problem (score nightly → hand high-risk segment to the campaign tool). A real-time-only API would signal a misread of the domain. We ship a batch CLI as the real path and a thin `/score` FastAPI endpoint as an explicit "here's the request/response pattern" demo. |
| **D3** | Modelling story | **Honest diagnosis + business value** | Given the 0.62 ceiling, intellectual honesty + a value story under uncertainty beats a fabricated headline number. We still do rigorous FE/tuning but **report negative results truthfully**. |
| **D4** | Tracking / registry | **Lightweight local registry** (versioned dir + `pipeline.joblib` + `metadata.json`), MLflow noted as prod swap-in | Reproducible + self-contained, zero infra. Standing up an MLflow server for this data size *is* the cargo-cult we're avoiding. |
| **D5** | Interpretability | **sklearn-native** — standardized LogReg coefficients as **odds ratios** + **`permutation_importance`** + per-customer **reason codes** from signed linear contributions | Perfect fit since the linear model wins; avoids the bias of RF impurity importance; no heavyweight SHAP dependency for a linear 0.62-AUC model. Reason codes feed `scored.csv`. |
| **D6** | CI/CD | **Full** — ruff + pytest (coverage gate) + pipeline smoke run + Docker build/publish + manual-gated deploy stage | Maximum surface to demonstrate the full lifecycle. Deploy stage is a documented, manually-gated placeholder (no live target). |

**Cleaning policy (committed default, D-clean):** repair impossible values → `NaN` → impute **inside the pipeline** (fit per fold); **winsorize** outliers (cap to train-fold quantiles) rather than delete rows. We **never drop rows from the test set** — deleting test records by an IQR rule computed across all data is itself leakage and silently changes the evaluation population.

---

## 3a. Implementation status (D1-D6 scope — built)

The leak-free, reproducible core is implemented and verified, and the lightweight
showcase layers from D1-D6 are now present: batch scoring, FastAPI demo,
monitoring report, Dockerfile, and CI workflow.

**Built & green:** `src/churn/{config,data,cleaning,features,pipeline,evaluate,interpret,registry,scoring,train,plots}.py`,
`src/churn/monitoring.py`, `api/serve.py`, the thin `churn_prediction.py` entrypoint,
`tests/` (40 tests passing), `pyproject.toml` + `uv.lock` (editable package,
`uv`-managed), `Dockerfile`, and `.github/workflows/ci.yml`. `uv run python churn_prediction.py`
runs end-to-end from a clean checkout, emitting 8 figures + the registered model.

**Measured result:** Logistic Regression is selected from a **statistical tie**
(CV ROC-AUC 0.594 ± 0.027 vs RF 0.592; paired hold-out AUC-gap CI [−0.041, +0.062]
straddles 0) via a tie-aware rule that prefers the simplest model — confirming the
thesis that complexity doesn't help. Calibrated hold-out AUC 0.644. `t*` = 0.51 at
the *baseline illustrative* economics; an EV sensitivity sweep shows targeting only
clearly beats blanket in the mid-cost regime (see WRITEUP Part 5).

**Codex findings → resolution:**

| Finding | Status | Where |
|---|---|---|
| #1 Leakage + AUC from hard labels | **Fixed** | all preprocessing inside `Pipeline`/`ColumnTransformer` fit per CV fold (`pipeline.py`); AUC from `predict_proba` (`evaluate.py`); guarded by `test_pipeline_leakage.py` + `test_metrics.py` |
| #2 Entrypoint can't find data | **Fixed** | paths resolved from repo root in `config.py`; `OneHotEncoder(handle_unknown="ignore")` replaces `LabelEncoder` |
| #3 Env not reproducible | **Fixed** | deps declared in `pyproject.toml` + `uv.lock`; smoke command `uv run python -m churn.train --smoke` |
| #4 Batch-relative risk tiers | **Fixed** | frozen cutpoints anchored to `t*`, persisted in the registry, shared `score_to_tier()`; `test_scoring_contract.py` proves batch-invariance |

**Adversarial-review round 2 (pre-submission) → resolution:**

| Finding | Status | Where |
|---|---|---|
| #1 EV headline rests on hand-picked economics | **Fixed** | config comment rewritten to flag the economics as *illustrative, not measured*; `COST_SCENARIOS` + `evaluate.threshold_sensitivity` add a real sweep; WRITEUP Part 5 relabels the dollar figure as conditional and shows targeting only clearly wins mid-regime |
| #2 "LogReg beats RF" on a 0.002 AUC gap inside noise | **Fixed** | `train._select_model` tie-aware rule (simplest within `MODEL_SELECTION_TOLERANCE`); `evaluate.bootstrap_auc_diff_ci` paired CI straddles 0; all docs reframed "wins" → "selected from a tie" |
| #3 Docker image can't score from a clean build | **Fixed** | `Dockerfile` trains + registers a model at build time (`RUN python -m churn.train`); CI runs the container and curls `/health` + `/score`; verified locally end-to-end |

---

## 4. Target architecture & repo layout

```
customer-attrition-prediction/
├── data/
│   └── customer_data.csv            # immutable input (do not modify)
├── churn_prediction.py              # THIN entrypoint: runs full exercise end-to-end,
│                                     #   emits all PNGs + prints all metrics. Satisfies
│                                     #   the README "complete and runnable" checklist.
├── src/churn/
│   ├── __init__.py
│   ├── config.py                    # paths, column groups, VALID_RANGES, cost params, SEED
│   ├── data.py                      # load + schema validation (training or scoring contract)
│   ├── cleaning.py                  # DomainRepair + Winsorizer transformers (train-fit)
│   ├── features.py                  # hypothesis-driven feature transformers
│   ├── pipeline.py                  # build_pipeline(model): ColumnTransformer + estimator
│   ├── train.py                     # repeated CV, modest search, calibration, threshold, persist
│   ├── evaluate.py                  # metric computation (P/R/F1/ROC-AUC/PR-AUC/Brier/$EV)
│   ├── plots.py                     # all figures (matplotlib Agg backend, savefig only)
│   ├── interpret.py                 # odds ratios, permutation importance, reason codes
│   ├── scoring.py                   # batch scoring CLI: raw csv → scored csv
│   ├── registry.py                  # save/load run dir + metadata.json (data hash, git sha, versions)
│   └── monitoring.py                # PSI/KS drift report vs a reference batch
├── api/
│   └── serve.py                     # FastAPI /score demo (loads latest registry model)
├── tests/
│   ├── test_data.py                 # schema validation and targetless scoring batches
│   ├── test_cleaning.py             # impossible→NaN, winsor caps fit on train only
│   ├── test_features.py             # feature transforms, no NaN leakage
│   ├── test_pipeline_leakage.py     # GUARD: scaler/imputer stats are train-only
│   ├── test_metrics.py              # metric correctness vs hand-derived values
│   ├── test_scoring_contract.py     # batch CLI I/O; probs∈[0,1]; tier monotonicity
│   ├── test_monitoring.py           # drift report + dirty-row counts
│   └── test_api.py                  # FastAPI demo contract
├── reports/
│   └── figures/                     # *.png deliverables
├── models/                          # generated registry artifacts (gitignored)
├── monitoring/
│   └── drift_report.md              # generated
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml                   # deps + [tool.ruff] + [tool.pytest] + coverage config
├── README.md                        # PROJECT readme: how to run (distinct from exercise README)
├── WRITEUP.md                       # the narrative deliverable (decisions + findings)
└── IMPLEMENTATION_PLAN.md           # this file
```

**Dependency set** (managed with `uv`):
- Core runtime: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.
- Serving extra: `fastapi`, `uvicorn` (with `pydantic` pulled transitively by FastAPI).
- Dev: `pytest`, `pytest-cov`, `ruff`, `httpx` (FastAPI test client). `mypy` optional.
- **No XGBoost/LightGBM/SHAP/MLflow** — deliberately. They would add weight and zero insight given the signal ceiling (D4, D5). Each omission is justified in WRITEUP, which is itself a senior signal.

**Design invariants**
- *Single source of truth for the transform.* The exact same fitted `Pipeline` object is used in training, batch scoring, and the API. No re-implemented preprocessing at inference → no train/serve skew.
- *Everything fittable lives inside the `Pipeline`.* Imputers, winsorizer, encoder, scaler, calibrator. CV wraps the whole pipeline. This is the structural fix for the starter's leakage.
- *Determinism.* One `SEED` in `config.py`; every estimator and split seeded; data SHA-256 recorded in metadata.

---

## 5. Cleaning & preprocessing policy (Part 2)

Implemented as custom, train-fit sklearn transformers so there is **no leakage** and the policy is unit-tested.

| Column | Valid range | Impossible-value action | Outlier action | Imputation (fit on train fold) |
|---|---|---|---|---|
| `account_age_days` | `[0, 1500]` | `<0` → `NaN` | none (no extreme tail) | median + indicator |
| `days_since_last_login` | `[0, 365]` | `<0` or `>365` (incl. `800/999` sentinels) → `NaN` | none | median + indicator |
| `monthly_spend` | `≥ 0` | — | **winsorize** to train `[p1, p99]` | median + indicator |
| `avg_order_value` | `≥ 0` | — | **winsorize** to train `[p1, p99]` | median + indicator |
| `pages_per_session` | `≥ 0` | — | none | median + indicator |
| `num_orders_last_90d` | `≥ 0` | none (see note ↓) | none | (rarely missing) median |
| `support_tickets_raised` | `[0, ~10]` | — | none | none observed |
| `region`/`device_type`/`subscription_plan` | finite sets | unseen-category-safe (`handle_unknown='ignore'`) | — | most-frequent (defensive) |

**Rejected "logical-consistency" rule** (`orders == 0` but spend `> 0`, 11 rows): an earlier version nulled the spend and emitted a `logically_inconsistent` flag, on the theory that zero orders should mean zero spend. That theory is wrong here: `monthly_spend` is a **6-month** average while `num_orders_last_90d` is a **90-day** count, so a customer who purchased in months 4–6 but not the last 90 days legitimately has spend with zero recent orders. The 11 affected rows are normal lapsing customers (real spend, varied tenure), and the flag was non-predictive (45.5% churn vs 48.4% base) and — once winsorized — silently constant. We **removed the rule**: spend is preserved, no flag is emitted. Genuinely impossible values are still repaired via `VALID_RANGES`.

**Why winsorize, not IQR-delete (the starter's approach)?**
1. Deleting rows by a global IQR rule **removes would-be test rows and changes the evaluation population** — methodologically invalid.
2. The injected spend spikes are *leverage* points; a scaled linear model is acutely sensitive to them. Capping neutralises the leverage while **retaining the row's other signal**.
3. Caps are learned on the **training fold only** (`Winsorizer.fit` stores train quantiles) → no leakage.

**Encoding & scaling**
- Categoricals → `OneHotEncoder(handle_unknown='ignore')` (not `LabelEncoder` — see §6 bug catalog).
- Numerics → `StandardScaler` (needed by LogReg; harmless for trees, kept for a single uniform pipeline).
- Both inside the `ColumnTransformer`.

---

## 6. Starter-code review (Part 3) — full defect catalog

The README hints "*some sections … have subtle logical issues*." There are **3 labelled bugs and ~11 unlabelled issues**. Catalog below; fixes land via the rewrite into `src/churn/` (the starter is superseded by the thin entrypoint + package, with the diagnosis preserved in WRITEUP).

**Labelled bugs**
1. **L56** `churn_by_plan.values.sort()` returns `None` (in-place sort) → `plt.bar` receives `None`; intended sorted bars with **aligned** labels are never produced. *Fix:* `s = churn_by_plan.sort_values(); plt.bar(s.index, s.values)`.
2. **L154** `roc_auc_score(y_test, preds)` passes **hard labels** instead of `proba` → AUC collapses to balanced-accuracy and discards all ranking. *Fix:* pass `predict_proba[:,1]`.
3. **L206** plots unsorted `importances` instead of `imp_sorted`. *Fix:* plot `imp_sorted` (and prefer permutation importance, see #11).

**Unlabelled issues (the senior-level catch)**
4. **Data leakage (critical).** IQR outlier removal (L73–77), median/mean imputation (L86–88), and `StandardScaler` (L128–130) are fit on the full dataset before/around the split. Test statistics bleed into training → optimistic, irreproducible metrics. *Fix:* all of it inside a `Pipeline`, fit per CV fold.
5. **Row deletion mutates the test set.** IQR filter drops records globally (§5). *Fix:* winsorize, never drop.
6. **`LabelEncoder` on nominal features** imposes a fake ordinal ranking on `region/device/plan` — wrong for LogReg, and `LabelEncoder` is intended for *targets*. Fitting on full data also breaks on unseen categories at inference. *Fix:* `OneHotEncoder(handle_unknown='ignore')`.
7. **No reusable fitted transform → train/serve skew risk.** Encoders/scaler are loose variables, not persisted as one object. *Fix:* single `Pipeline` persisted to the registry; reused verbatim at scoring.
8. **`plt.show()` in a batch script** blocks / needs a display → not CI-runnable. *Fix:* `matplotlib.use("Agg")`, `savefig` only.
9. **Hard-coded relative path** `../data/customer_data.csv` assumes cwd = `starter_code/`; repo has the script at root → `FileNotFoundError` when run as documented. *Fix:* resolve paths from `config.py` / project root.
10. **Single 80/20 split on 1,600 rows** → high-variance estimates. *Fix:* `RepeatedStratifiedKFold` (5×3) for model selection; one held-out test split for the README's headline metrics + plots.
11. **RF impurity `feature_importances_`** is biased toward high-cardinality/continuous features. *Fix:* `permutation_importance` on the validation set.
12. **RF hard-coded as "best model"** for the confusion matrix with no comparison. *Fix:* select by CV ROC-AUC with a **tie-aware rule** (prefer the simplest model within noise; paired bootstrap CI documents the tie → LogReg) and plot *its* matrix at the **business threshold**, not 0.5.
13. **Reflexive imbalance assumption.** The target is balanced; the imbalance is in `subscription_plan`. *Fix:* no resampling; document why. (The starter doesn't resample — but a candidate "improving" it often adds SMOTE here. We explicitly won't.)
14. **Feature semantics.** `spend_per_order = monthly_spend/(orders+1)` divides a **6-month** spend by a **90-day** order count (unit mismatch); `engagement_score` mixes recency into a per-session metric. *Fix:* drop/redefine with correct units (§7); test each for incremental lift.

---

## 7. Feature engineering (Part 2.5) — hypothesis-driven, honestly reported

Given the signal ceiling, FE is framed as **hypotheses, each tested for incremental CV-AUC lift**; survivors kept, the rest documented as **negative results** (a senior signal in itself).

| Candidate | Hypothesis | Keep criterion |
|---|---|---|
| `recency_ratio = days_since_last_login / (account_age_days+1)` | inactivity *relative to tenure* > absolute recency | +ΔAUC in CV |
| `is_dormant = days_since_last_login > 90` | step-change in disengagement | +ΔAUC |
| `support_per_order = support_tickets / (orders+1)` | friction per unit of activity | +ΔAUC |
| missingness indicators | informative missingness (MAR) | +ΔAUC |

> The implemented feature set keeps `recency_ratio`, `is_dormant`, `support_per_order`,
> and missingness indicators. The write-up reports that these add only marginal lift
> beyond plan + recency (all land near an odds ratio of 1.0) instead of inflating their impact.

---

## 8. Modelling & evaluation protocol (Part 4)

**Models (≥2 required; we present 3, select by CV):**
- `LogisticRegression` (L2, `C` tuned) — the expected winner.
- `RandomForestClassifier` — README's nonlinear reference.
- `HistGradientBoostingClassifier` — sklearn-native GBM challenger (no extra dep).

**Validation**
- **Model selection:** `RepeatedStratifiedKFold(5×3)` → report mean ± std ROC-AUC (stable on 1,600 rows).
- **Headline metrics + plots:** one stratified 80/20 hold-out, evaluated **once** at the end (satisfies the README's "on the test set").
- **Hyperparameters:** deliberately modest fixed candidates; chasing decimals on a 0.62 ceiling is theatre.

**Calibration (core to the value story, D3)**
- Wrap the selected model in `CalibratedClassifierCV` with sigmoid calibration.
- Report **Brier score** + a **reliability diagram** (before/after). We are selling *probabilities*, so they must be trustworthy.

**Decision threshold — cost-based, not 0.5**
- Choose `t*` to **maximise expected campaign value** on out-of-fold validation probabilities (formula in §9), then report Precision/Recall/F1 **at `t*`**.
- Confusion matrix for the best model is plotted **at `t*`**; 0.5 metrics are printed for contrast.

**Risk tiers — frozen cutpoints, not batch-relative** *(resolves Codex #4)*
- Tiers are defined by **fixed probability cutpoints** anchored to the business threshold:
  `high ≥ t*` (campaign-worthy), `medium ≥ t_mid`, else `low`, where `t_mid` is the
  median of the validation scores below `t*`.
- Both cutpoints are **frozen at train time and persisted** in `metadata.json`, and a
  single `score_to_tier()` is shared by the batch CLI and the API. A customer's tier
  therefore depends only on their own probability — never on who else is in the batch.
  Enforced by `tests/test_scoring_contract.py` (batch-invariance + single-record parity).
- The cutpoints track the economics: the break-even is `t_be = c_contact / (V_save·u)`,
  so cheap-contact regimes push `t*` down (target broadly) and expensive ones push it up.

**Metrics reported per model**
- README-required: **Precision, Recall, F1, ROC-AUC**.
- Added: **PR-AUC** (more informative for a targeting decision), **Brier** (calibration), **expected $ value at `t*`**.
- **ROC curves** for both/all models on one axes (+ random baseline); **confusion matrix** for the best model.

**"Which metric matters most?" (Part 4.5 answer, pre-committed)**
> *Model selection:* ROC-AUC (threshold-independent). *Operating point:* **Recall**, subject to a precision/budget constraint — a missed churner forfeits their CLV, which typically dwarfs the cost of one wasted retention contact. But unbounded recall blows the campaign budget on false positives, so the real objective is **expected $ value**, which is exactly what `t*` optimises. We lead with $EV and explain P/R/F1 as its components.

---

## 9. Business interpretation & value model (Part 5)

**Highest-risk profile** (from coefficients + segment rates): **dormant** customers (high `days_since_last_login`) on the **Free** plan, secondarily lower-tenure. Quantified via odds ratios and segment churn rates.

**Recommended retention action:** a **win-back / re-engagement** play for dormant Free users (re-engagement email + a time-boxed plan-upgrade incentive), **prioritised by expected value** `p_churn × CLV`, not by raw probability — concentrating spend where it saves the most margin.

**Value model (turns a 0.62 model into dollars).** For a campaign at threshold `t`:

```
contacted        = TP + FP                      # customers with p ≥ t
EV(t) = TP · u · V_save  −  (TP + FP) · c_contact
   where  u        = campaign uplift (fraction of true churners actually retained)
          V_save   = margin saved by retaining one churner (≈ monthly_spend × expected horizon)
          c_contact= cost per outreach
```

We pick `t* = argmax EV(t)` on validation, present a **sensitivity table** over `(u, V_save, c_contact)`, and report the model's lift as **$ saved vs. (a) no campaign and (b) contact-everyone**. This reframes "AUC 0.62" as "$X recovered per 1,000 customers at the optimal operating point."

**Measuring real-world impact (Part 5.3):** a **randomised holdout** — model-targeted vs. control among high-risk customers — measuring **incremental retention (uplift)**, not raw retention, plus targeting precision and realised $ saved. Track these on the live score distribution over time (§11).

---

## 10. Serving (D2): batch-first + REST demo

**Primary — batch scoring CLI** (`python -m churn.scoring --in data.csv --out scored.csv`):
- Validates input against the data contract, applies the persisted pipeline, emits:
  `customer_id, churn_probability, risk_tier, top_reason_codes`.
- `risk_tier` uses the **frozen cutpoints** from §8 (`high ≥ t*`, `medium ≥ t_mid`),
  loaded from the registry — identical mapping in batch and API, never batch-relative.
- Reason codes = top signed `coef × z(feature)` contributions per customer (from `interpret.py`),
  computed from the persisted uncalibrated linear base (calibration is monotone, so driver
  ranking is preserved).

**Demo — FastAPI `/score`** (`api/serve.py`): single-record endpoint, same persisted pipeline + same `score_to_tier()`, Pydantic-validated request, returns `{churn_probability, risk_tier, reason_codes}`. A pattern demonstration, not the production path.

---

## 11. Monitoring & drift (lightweight, D1+D4)

`monitoring.py` produces `monitoring/drift_report.md` comparing a **reference** batch (training data) to a **current** batch:
- **Input drift:** PSI per feature + KS test (flag PSI > 0.2).
- **Prediction drift:** score-distribution shift (PSI on `churn_probability`).
- **Data-quality drift:** reuse the §5 data contract — schema, range, missingness-rate checks.
- **Performance drift:** documented cadence — labels arrive 90 days later, so performance is recomputed on a lag; the report templates the table and alert thresholds.

---

## 12. Testing strategy (D6)

| Test file | Guards |
|---|---|
| `test_data.py` | schema present, targetless scoring batches, fail-fast on bad input |
| `test_cleaning.py` | impossible→NaN mapping; **winsor caps fit on train only** (re-fit with test rows changes caps) |
| `test_features.py` | transforms produce no NaN leak; `recency_ratio` bounded; flags boolean |
| `test_pipeline_leakage.py` | **leakage guard:** scaler/imputer statistics equal train-only statistics; whole pipeline sits inside CV |
| `test_metrics.py` | P/R/F1/ROC-AUC match hand-derived values on a toy set (catches the L154-class bug) |
| `test_scoring_contract.py` | CLI round-trip; `churn_probability ∈ [0,1]`; `risk_tier` monotone in probability |
| `test_monitoring.py` | dirty-row counts, PSI stability, markdown report generation |
| `test_api.py` | FastAPI `/health` + `/score` against a registered model |

Coverage gate on `src/churn` core (≥ 80%).

---

## 13. CI/CD & Docker (D6, full)

**`.github/workflows/ci.yml`**
1. `ruff check` + format check.
2. `pytest --cov` with coverage gate.
3. **Pipeline smoke run** on a sampled slice (`python -m churn.train --smoke`) — proves end-to-end runnability.
4. **Docker build** of the API image, then a **container smoke test** — run the image and curl `/health` + `/score` to prove the deployed artifact actually scores from a clean build (not just that it builds); publish to GHCR on `main`.
5. **Deploy** stage — manually-gated placeholder (no live target), documented as the prod hook.

**`Dockerfile`:** slim Python base, deps via `uv`, **trains + registers a model at build time** so the image is self-contained (a bare `docker run` serves `/score` with no mounted volume), `uvicorn` entrypoint for the API; the same image runs the batch CLI.

---

## 14. Execution phases (mapped to README)

| Phase | Output | README part |
|---|---|---|
| **0. Scaffold** | repo layout, `pyproject.toml` (deps + ruff + pytest), `config.py`, `uv` env, Agg backend | setup |
| **1. Contract + EDA** | `data.py`, profiling, EDA figures (target balance, churn-by-plan/region, behavioural distributions by class), imbalance commentary | **Part 1** |
| **2. Cleaning + features** | `data.py`, `cleaning.py`, `features.py` (+ tests); impossible-value repair, winsorize, impute, encode/scale, ≥1 engineered feature | **Part 2** |
| **3. Starter review** | bug catalog (§6) in WRITEUP; starter superseded by package | **Part 3** |
| **4. Modelling** | `pipeline.py`, `train.py`, `evaluate.py`, `plots.py`; 3 models, repeated CV, calibration, cost-based threshold; ROC + confusion + calibration PNGs; metrics table | **Part 4** |
| **5. Interpretation** | `interpret.py`; odds ratios, permutation importance, reason codes; risk profile, retention action, impact-measurement plan | **Part 5** |
| **6. Serving** | `scoring.py` (batch CLI), `api/serve.py` (demo), `registry.py` | D2/D4 |
| **7. Monitoring** | `monitoring.py` + `drift_report.md` | D1 |
| **8. Tests + CI + Docker** | `tests/`, `ci.yml`, `Dockerfile` | D6 |
| **9. Writeup + entrypoint** | `churn_prediction.py` (runs 1→5 end-to-end), `WRITEUP.md`, project `README.md` | submission |

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Over-engineering optics** for 1,800 rows | Every component carries a domain/eval justification; heavy infra (MLflow/SHAP/XGBoost) deliberately omitted *and the omission defended* (D4/D5). Lightweight by design. |
| **Weak signal disappoints a naive reader** | Lead with the honest diagnosis + the $-value reframing (D3, §9). Set expectations in the executive summary. |
| **Reviewer expects high AUC** | WRITEUP pre-empts: shows the signal audit, why complexity didn't help, and what richer data (clickstream, payment events, NPS) would lift it. |
| **Scope creep across 9 phases** | Phases are vertically sliced; the package is usable after Phase 5 (the exercise core); 6–8 are additive showcase layers that can't break the core. |
| **Leakage reintroduced during edits** | `test_pipeline_leakage.py` is a standing guard in CI. |

---

## 16. Submission checklist (README) → coverage

- [x] `churn_prediction.py` complete & runnable end-to-end → thin entrypoint orchestrating the package.
- [x] All plots saved as `.png` → `reports/figures/` (Agg backend, no `plt.show`).
- [x] `WRITEUP.md` covering decisions & findings → narrative built around the §1 thesis, §3 decision log, §6 bug catalog, §9 business model.
- [x] Part 1 EDA · Part 2 cleaning/FE · Part 3 starter fixes · Part 4 two+ models + metrics + ROC + confusion · Part 5 business interpretation — all mapped in §14.

---

## 17. Explicitly out of scope

Real-time feature store; streaming inference; distributed training (unjustifiable at this scale); SHAP/XGBoost/LightGBM/MLflow (omitted by decision, defended in WRITEUP); live deployment target (deploy stage is a documented placeholder).
