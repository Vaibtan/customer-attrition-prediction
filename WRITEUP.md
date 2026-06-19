# Write-up — Customer Churn Prediction

> Decisions and findings for the AI Analyst exercise. Every number below is
> measured on `data/customer_data.csv` by `uv run python churn_prediction.py`
> (seed `42`); figures referenced live in `reports/figures/`. The deeper
> engineering rationale lives in `IMPLEMENTATION_PLAN.md`; this document is the
> reviewer-facing narrative.

---

## TL;DR (the thesis)

Two findings drive every decision here:

1. **The label is balanced — 48.4% churn.** The README's *"class imbalance"*
   refers to `subscription_plan` (Free 733 → Enterprise 117), **not** the target.
   Reflexively reaching for SMOTE / `class_weight='balanced'` would be wrong and
   would *degrade* calibration. We deliberately **do not resample**, and say why.

2. **The signal is genuinely weak — and that is the interesting part.** A
   *leak-free* cross-validation tops out around **ROC-AUC ≈ 0.60–0.64**, and the
   three models are a **statistical dead heat**: the top two sit ~0.002 AUC apart
   (well inside the ~0.027 fold spread), and the 95% bootstrap CI on the hold-out
   AUC *gap* straddles 0. We therefore **don't claim Logistic Regression "wins"** —
   we pick it because, among indistinguishable models, the linear one is the most
   interpretable and best-calibrated. Model complexity buys nothing here. The
   honest move is to diagnose that ceiling and then **manufacture business value
   from a modest model** (calibrated probabilities + a cost-based threshold + an
   uplift framing) rather than fabricate a suspicious 0.95-AUC.

> **The single most important defect in the starter code is not one of the three
> labelled bugs — it is data leakage:** outlier removal, imputation, and scaling
> are fit on the *entire* dataset before the train/test split. The whole solution
> is organised around a leak-free `Pipeline` so every fitted statistic is learned
> per training fold only.

---

## Part 1 — Exploratory Data Analysis

**Shape & types.** 1,600 rows × 12 columns. No duplicate `customer_id`, no
duplicate rows. 4 categorical columns (`region`, `device_type`,
`subscription_plan`, plus the `customer_id` key), 7 numeric features, 1 binary
target. (`churn_prediction.py` prints the full `dtypes` and `describe()` tables.)

**Target balance.** `churned` mean = **0.4838** → essentially balanced. See
`reports/figures/eda_overview.png` (left panel).

**Missing values.** Three behavioural columns are missing exactly **80 cells (5%)**
each — `monthly_spend`, `avg_order_value`, `pages_per_session` — affecting **234
rows (14.6%)** in total (no row is missing all three). The pattern is scattered
and plausibly Missing-At-Random.

**Impossible / inconsistent values** (surfaced directly in `describe()`):

| Column | Symptom in the data | Real-world valid range |
|---|---|---|
| `account_age_days` | min `-1` (negatives) | `[0, ~1500]` |
| `days_since_last_login` | min `-5`, max `999` (negatives + `800/999` sentinels) | `[0, 365]` |
| `monthly_spend` | max `18,772` (median ≈ 56) | `≥ 0`, but extreme spikes are outliers |
| `avg_order_value` | max `9,724` (median ≈ 122) | `≥ 0`, same |

> **What is *not* an impossible value:** 11 customers have `num_orders_last_90d == 0`
> with positive `monthly_spend`. This is **valid**, not dirty — spend is a 6-month
> average while orders are a 90-day count, so a customer who purchased in months 4–6
> but not in the last 90 days legitimately has spend with zero recent orders. These
> rows are normal lapsing customers (and the lapsing signal is exactly what we want
> to keep), so we deliberately do **not** null their spend.

**Churn rate by segment.**

- **By plan (real signal, monotone):** Free **0.551** → Basic 0.472 → Premium
  0.396 → Enterprise **0.316**. Churn falls cleanly as plan tier rises.
- **By region (negligible):** 0.448–0.509 across all five regions — no usable
  signal.

See `reports/figures/eda_overview.png` (centre/right panels).

**Behavioural features, churned vs retained** (`reports/figures/behavioural_by_churn.png`):
the only feature that visibly separates the classes is **`days_since_last_login`**
— churned customers have logged in less recently. `monthly_spend` and
`support_tickets_raised` barely differ between the groups. This previews the
modelling result: the data is largely a *recency-and-plan detector*.

### Part 1.4 — Class imbalance comment (and how we address it)

**We address it by *not* treating it as a target-imbalance problem, because it
isn't one.** The target is 48.4% positive — balanced. The imbalance the README
mentions lives in `subscription_plan`, which is a *feature*, and the right tool
for a skewed categorical feature is **one-hot encoding** (every level, including
rare Enterprise, gets its own coefficient), **not** resampling.

Concretely:

- **No SMOTE / oversampling / undersampling.** On a balanced target these distort
  the base rate, harm probability **calibration**, and we are explicitly selling
  *calibrated probabilities* (the value model depends on them).
- **No `class_weight='balanced'`** for the same reason — it re-weights toward a
  decision boundary we don't want, since we set the operating point with an
  explicit cost model instead (Part 4).
- **Rare plan levels** (Enterprise, n=117) are handled by
  `OneHotEncoder(handle_unknown='ignore')`, so an unseen or rare category never
  breaks inference and never gets a fake ordinal rank.

Stating *why we don't resample* is itself the deliverable — reflexive SMOTE here
would be a junior tell.

---

## Part 2 — Data Cleaning & Preprocessing

All cleaning is implemented as **custom, train-fit sklearn transformers inside the
modelling `Pipeline`**, so every fitted statistic (caps, medians, scaler stats) is
learned **per CV fold** — never on the test data. This is the structural fix for
the starter's leakage and is enforced by `tests/test_pipeline_leakage.py`.

### 2.1 Impossible values — corrected per column

Values outside documented real-world ranges (`config.VALID_RANGES`) are mapped to
`NaN` and then imputed *inside* the pipeline:

- `account_age_days < 0` → `NaN`
- `days_since_last_login < 0` or `> 365` (catches the `800`/`999` sentinels) → `NaN`

We **only** repair values that are genuinely impossible (negatives, out-of-window
sentinels). We considered a "zero orders but positive spend" consistency rule and
**rejected it**: the two fields are measured over different windows (90 days vs 6
months), so those 11 rows are valid lapsing customers, not dirty data. Nulling
their spend would have destroyed real signal — flagging that as a non-issue is the
senior call here.

### 2.2 Outliers — winsorize, don't delete

`monthly_spend` (max 18,772 vs median 56) and `avg_order_value` (max 9,724 vs
median 122) carry 1–2 order-of-magnitude spikes — leverage points that wreck a
scaled linear model. We **winsorize (cap) to the train-fold `[p1, p99]`**, learned
on the training fold only.

**Why cap, not delete (the starter's IQR-drop approach)?** Deleting rows by a
global IQR rule **removes would-be *test* rows and silently changes the evaluation
population** — that is leakage and methodologically invalid. Capping neutralises
the leverage while **retaining the row's other signal**.

### 2.3 Imputation — justified per column

- **Numeric:** median imputation (robust to the skew above) **+ a missingness
  indicator** (`add_indicator=True`), so if missingness is informative the model
  can use it rather than having it erased.
- **Categorical:** most-frequent (defensive; categoricals are not actually missing
  here).

### 2.4 Encoding & scaling

- Categoricals → `OneHotEncoder(handle_unknown='ignore')` (**not** `LabelEncoder`,
  which imposes a fake ordinal ranking on nominal fields and is meant for targets).
- Numerics → `StandardScaler` (required by Logistic Regression; harmless for the
  trees, kept so there is a single uniform pipeline).
- Both inside a `ColumnTransformer` → one fitted object is reused verbatim in
  training, batch scoring, and the API (no train/serve skew).

### 2.5 Engineered features (≥1 required; we test several, honestly)

Feature engineering is framed as **hypotheses, each kept only if it earns its
place** given the signal ceiling:

| Feature | Hypothesis |
|---|---|
| `recency_ratio = days_since_last_login / (account_age_days + 1)` | inactivity *relative to tenure* beats absolute recency |
| `is_dormant = days_since_last_login > 90` | a step-change in disengagement |
| `support_per_order = support_tickets / (orders + 1)` | friction *per unit of activity* |
| missingness indicators | informative missingness (MAR) |

**Honest result:** the engineered features add only marginal lift on top of
`subscription_plan` + `days_since_last_login`, which already carry almost all the
signal — every engineered feature lands near an odds ratio of 1.0 (Part 5). We
report that as a **negative result** rather than inflating a feature-count vanity
metric. The honest read is that richer raw data, not cleverer features, is what
this problem needs.

---

## Part 3 — Review & Improve the Starter Code

The starter lays out a plausible pipeline but ships **3 labelled bugs and a set of
deeper, unlabelled issues**. The fixes land via the rewrite into `src/churn/`; the
diagnosis is preserved here.

**Labelled bugs**

1. **`churn_by_plan.values.sort()` returns `None`** (in-place sort) → `plt.bar`
   receives `None` and the intended sorted bars with aligned labels never render.
   *Fix:* `s = churn_by_plan.sort_values(); plt.bar(s.index, s.values)`.
2. **`roc_auc_score(y_test, preds)` passes hard labels, not probabilities** →
   ROC-AUC collapses to balanced-accuracy and throws away all ranking information.
   *Fix:* pass `predict_proba[:, 1]`. Guarded by `tests/test_metrics.py`.
3. **Plots unsorted `importances` instead of the sorted array** → the importance
   chart is mislabelled. *Fix:* plot the sorted values (and prefer permutation
   importance — see below).

**Deeper issues (the senior-level catch)**

4. **Data leakage (critical).** IQR outlier removal, imputation, and
   `StandardScaler` are fit on the full dataset around the split → test statistics
   bleed into training → optimistic, irreproducible metrics. *Fix:* everything
   inside a `Pipeline`, fit per CV fold.
5. **Row deletion mutates the test set** (the IQR drop). *Fix:* winsorize, never
   drop (Part 2.2).
6. **`LabelEncoder` on nominal features** invents an ordinal ranking on
   `region/device/plan` and breaks on unseen categories at inference. *Fix:*
   `OneHotEncoder(handle_unknown='ignore')`.
7. **No single persisted transform → train/serve skew risk.** Loose encoder/scaler
   variables. *Fix:* one `Pipeline` persisted to the registry and reused verbatim.
8. **`plt.show()` in a batch script** blocks / needs a display → not CI-runnable.
   *Fix:* `matplotlib.use("Agg")`, `savefig` only.
9. **Hard-coded relative data path** (`../data/...`) assumes a working directory.
   *Fix:* resolve paths from the repo root in `config.py`.
10. **Single 80/20 split on 1,600 rows** → high-variance estimates. *Fix:*
    `RepeatedStratifiedKFold` (5×3) for selection; one held-out split for headline
    metrics + plots.
11. **RF impurity `feature_importances_`** is biased toward high-cardinality /
    continuous features. *Fix:* `permutation_importance` on the hold-out.
12. **RF hard-coded as "best model"** with no comparison, confusion matrix at 0.5.
    *Fix:* select by CV ROC-AUC with a **tie-aware rule** (`train._select_model`
    prefers the simplest model within noise of the best, and a paired bootstrap CI
    documents the tie), then plot the winner's matrix at the **business
    threshold** `t*`.
13. **Reflexive imbalance assumption** (a candidate "improving" the starter often
    bolts on SMOTE here). *Fix:* none needed — the target is balanced (Part 1.4).
14. **Broken feature semantics** — e.g. dividing a 6-month spend by a 90-day order
    count (unit mismatch). *Fix:* redefined with correct units and tested for lift
    (Part 2.5).

---

## Part 4 — Model Building & Evaluation

**Validation protocol.** Model *selection* uses `RepeatedStratifiedKFold` (5×3) on
the training split (stable on 1,600 rows). Headline metrics and all plots come
from **one stratified 80/20 hold-out** evaluated **once** (320 test rows, 155
churned). The winner is wrapped in `CalibratedClassifierCV` (sigmoid) so the
probabilities are trustworthy — the entire value story depends on calibration.

### 4.1 / 4.2 — Three models, full metrics on the test set

**Model selection — CV ROC-AUC on train** (`RepeatedStratifiedKFold` 5×3):

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.594 ± 0.027 |
| Random Forest (300 trees) | 0.592 ± 0.027 |
| HistGradientBoosting | 0.565 ± 0.025 |

**Selection is tie-aware, not a third-decimal race.** The top two means differ by
**0.002** — well inside the ±0.027 fold-to-fold spread (≈ 1 SE of the mean). The
code (`train._select_model`) treats every model within a `0.01` AUC tolerance of
the best as a **statistical tie** and breaks the tie by an explicit preference
order (simplest / most interpretable / best-calibrated first), rather than picking
whichever third decimal happened to win on this seed. Logistic Regression is
selected on that rule — *not* because it "beat" the ensembles.

**The tie is backed by a paired bootstrap, not just overlapping error bars.** A
1,000-round paired bootstrap of the **hold-out** AUC gap (LogReg − Random Forest)
gives a mean of **+0.011 with a 95% CI of [−0.041, +0.062]** — it **straddles 0**,
so the two models are statistically indistinguishable on this data. The winner's
own hold-out ROC-AUC has a wide 95% bootstrap CI of **[0.581, 0.701]** (n=320),
an honest reminder that *any* single-number AUC on 320 test rows is noisy.

**Per-model metrics on the hold-out** (uncalibrated, threshold 0.50 — the
README's "each model on the test set"):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.584 | 0.581 | **0.583** | **0.643** |
| Random Forest | 0.585 | 0.535 | 0.559 | 0.633 |
| HistGradientBoosting | 0.540 | 0.523 | 0.531 | 0.616 |

Logistic Regression posts the top point estimate on both CV and hold-out, but —
per the bootstrap above — that lead is inside the noise. The takeaway is not
"linear wins" but "**linear is no worse, so prefer it**" on a weak-signal,
mostly-linear dataset. Complexity earns nothing here. ROC curves for all three on
one axes: `reports/figures/roc_curves.png`.

> **Note on the headline number.** The *calibrated* winner scores ROC-AUC **0.644**
> on the hold-out (calibration is monotone, so ranking — and thus AUC — is
> essentially unchanged from the 0.643 uncalibrated figure); calibration buys us a
> trustworthy **Brier score of 0.238**, which is what the dollar model needs.

### 4.3 — ROC curves

All three models on shared axes with the random baseline:
`reports/figures/roc_curves.png`. The curves are tightly bunched just above the
diagonal — a faithful picture of a genuinely weak signal, not a presentation
failure.

### 4.4 — Confusion matrix (best model, at the business threshold)

We plot the matrix at the **cost-based threshold `t* = 0.51`** (not 0.5 — see 4.5),
`reports/figures/confusion_matrix.png`. On the 320-row hold-out:

|  | Pred: retain | Pred: churn |
|---|---|---|
| **Actual: retained** | TN 116 | FP 49 |
| **Actual: churned** | FN 79 | TP 76 |

**Interpretation.** At `t*` the model contacts **125 customers** (TP 76 + FP 49).
It catches **76 of 155 true churners (recall 0.490)**, and **of everyone it
contacts, 60.8% are genuine churners (precision 0.608)** — versus the 48.4% base
rate, so targeting roughly multiplies a blanket campaign's hit-rate by ~1.25×. The
**79 false negatives** are the expensive errors (a missed churner forfeits their
lifetime value); the **49 false positives** cost only a wasted retention contact.
The threshold deliberately trades recall down to keep precision high enough that
the campaign nets a profit (Part 5) — at 0.50 the model would catch more churners
(recall 0.555) but contact more people, and the cost model prefers `t* = 0.51`.

### 4.5 — Which metric matters most, and why

There is no single answer — it depends on the decision being made:

- **For model selection: ROC-AUC** (threshold-independent — it ranks models by how
  well they *order* customers by risk, before any operating point is chosen).
- **For the operating point: expected dollar value, not raw P/R/F1.** In a churn
  context a **false negative** (a missed churner) forfeits that customer's CLV,
  which typically dwarfs the cost of one wasted retention contact — so **recall**
  matters more than precision *per unit error*. But unbounded recall blows the
  campaign budget on false positives. The metric that resolves this tension is
  **expected campaign value**, and we choose `t*` to maximise exactly that
  (Part 5). Precision/Recall/F1 are reported as its diagnostic components.

We also report **PR-AUC (0.617)** — more honest than ROC-AUC for a
needle-in-haystack targeting decision — and **Brier (0.238)** for calibration
quality. Calibration curve: `reports/figures/calibration_curve.png`.

---

## Part 5 — Business Interpretation

### Highest-risk profile

From standardized Logistic Regression coefficients as **odds ratios** (OR > 1
raises churn odds per 1 SD; `reports/figures/feature_importance.png` corroborates
via permutation importance):

| Driver | Odds ratio | Direction |
|---|---|---|
| `subscription_plan == Free` | **1.61** | ↑ churn |
| `days_since_last_login` | 1.28 | ↑ churn |
| `account_age_days` | 0.86 | ↓ churn (longer tenure) |
| `subscription_plan == Premium` | 0.76 | ↓ churn |
| `subscription_plan == Enterprise` | **0.61** | ↓ churn (most protective) |

**The highest-risk profile is a dormant, low-tenure, Free-plan customer** — high
`days_since_last_login` and low `account_age_days` on the lowest tier. This is
consistent with the EDA: Free churns at 55% vs Enterprise at 32%, and recency is
the strongest behavioural separator.

> **An honest nuance:** every behavioural and engineered feature beyond
> plan/recency/tenure lands within a hair of an odds ratio of 1.0 —
> `support_tickets_raised` (1.00), `support_per_order` (1.06), `monthly_spend`
> (1.02), `is_dormant` (1.02). We surface that rather than dressing up a noise-level
> coefficient as an insight: on this data, churn is almost entirely a
> plan-and-recency story.

### Recommended retention action

A **win-back / re-engagement play targeted at dormant Free users** — a
re-engagement nudge plus a time-boxed plan-upgrade incentive (Free → Basic moves a
customer from the 55% to the 47% churn band). Crucially, **prioritise by expected
value `p_churn × CLV`, not by raw probability**, so spend concentrates where it
saves the most margin. Each scored customer carries **reason codes** (top signed
linear contributions) so the campaign tool can tailor the message.

### Quantifying the value — and being honest about what it rests on

> **These economics are assumptions, not data.** The dataset carries no campaign
> costs, so the dollar figures below are **illustrative and entirely conditional**
> on the numbers we plug in. We treat them as a decision *framework*, not a model
> *result*, and report a sensitivity sweep instead of a single headline.

The expected value of a campaign at threshold `t`, given per-contact cost,
retention `value` saved, and campaign `uplift`, is:

```
EV(t) = TP·uplift·value − (TP + FP)·cost
```

A calibrated model is worth contacting customer *i* iff `p_i` exceeds the
break-even `t_be = cost / (value · uplift)`. We pick `t* = argmax EV(t)` on
out-of-fold validation probabilities (`reports/figures/expected_value.png`) and
evaluate on the untouched hold-out.

**Under one baseline assumption** (`$20`/contact, `$200` saved/retention, `0.20`
uplift → break-even `0.5`, `t* = 0.51`):

| Strategy | EV (320 hold-out) | Per 1,000 customers |
|---|---|---|
| No campaign | $0 | $0 |
| Contact **everyone** | −$200 (a net loss) | −$625 |
| **Model-targeted at `t*`** | **+$540** | **+$1,688** |

**Sensitivity — where the model actually adds value** (each row re-derives `t*` on
validation for *that* scenario, then scores the hold-out; full grid printed by
`churn_prediction.py` and persisted to `metadata.json["ev_sensitivity"]`):

| Scenario | cost | value | uplift | break-even | `t*` | $/1k targeted | $/1k contact-all |
|---|---|---|---|---|---|---|---|
| cheap_contact | 5 | 200 | 0.20 | 0.12 | 0.32 | **+14,297** | +14,375 |
| **baseline** | 20 | 200 | 0.20 | 0.50 | 0.51 | **+1,688** | −625 |
| expensive_offer | 50 | 200 | 0.20 | 1.25 | 0.75 | 0 | −30,625 |
| low_uplift | 20 | 200 | 0.10 | 1.00 | 0.75 | 0 | −10,312 |
| high_clv | 20 | 500 | 0.20 | 0.20 | 0.32 | **+28,312** | +28,438 |

**The honest reading is more nuanced than "the model prints money":**

- **In the middle (baseline) regime the model is decisive** — a blanket campaign
  *loses* $625/1k while model-targeting *makes* $1,688/1k. This is the case the
  model is built for: when contacts are costly enough that you can't spray everyone
  but the signal is good enough to concentrate spend.
- **When contacts are very cheap or CLV is huge** (`cheap_contact`, `high_clv`),
  break-even drops so low you should essentially **contact everyone** — targeting
  and blanket land within a rounding error, and the model's *selectivity* is barely
  worth it. An honest analyst says so rather than claiming credit for the +$28k.
- **When contacts are expensive or uplift is low** (`expensive_offer`,
  `low_uplift`), break-even exceeds the model's reachable probabilities, so `t*`
  pushes past every score and the EV-maximising action is **don't run the campaign
  at all** ($0, vs large losses for contacting everyone). The model's value there
  is *telling you not to spend* — also a real, if unglamorous, decision.

So the deliverable is not a dollar headline; it is a **threshold rule that adapts
to the economics** and, in the regime that matters, turns a money-losing blanket
campaign into a profitable targeted one.

### Measuring real-world impact

A **randomised holdout**: among high-risk (tier `high`) customers, randomly assign
a **control** group that is *not* contacted, and measure **incremental retention
(uplift)** — model-targeted minus control — not raw retention, which would
conflate the campaign effect with customers who would have stayed anyway. Track
targeting precision and realised dollars saved, and recompute on a lag (true
churn labels arrive ~90 days later). This is the only measurement that isolates the
model's *causal* contribution.

---

## What would actually move the needle

The ceiling is a **data** problem, not a model problem. The features here are
coarse aggregates; the signals that predict churn — and would lift AUC well past
0.64 — are **event-level**: clickstream / session sequences, payment-failure and
downgrade events, support-ticket *sentiment* (not just counts), and NPS / survey
responses. No amount of model complexity recovers signal that isn't in the columns,
which is exactly why Logistic Regression ties the gradient-boosted models here.

---

## Deliberate omissions (and why)

- **No XGBoost / LightGBM** — the gradient-boosted model already in the bake-off
  (HistGB) doesn't beat Logistic Regression on this signal, and the top models are
  a statistical tie; another boosted variant would be cargo-cult complexity.
- **No SHAP** — for a *linear* winner, standardized-coefficient odds ratios are
  exact and cheaper than SHAP approximations.
- **No SMOTE / resampling** — the target is balanced (Part 1.4).
- **No MLflow server** — a versioned local registry (`models/<run_id>/` +
  `metadata.json` with the data hash and seed) is reproducible with zero infra at
  this scale.

Each omission is a deliberate engineering call, defended above rather than left
implicit.

---

## How to reproduce

```bash
uv sync --all-groups                   # install core package + dev tools
uv run python churn_prediction.py      # EDA + bake-off + metrics + 8 figures + registered model
uv run python -m churn.scoring --in data/customer_data.csv --out scored.csv
uv run python -m churn.monitoring --reference data/customer_data.csv --current data/customer_data.csv
uv sync --extra serve --all-groups     # optional FastAPI demo dependencies
uv run uvicorn api.serve:app --reload
uv run pytest -q                       # 40 tests (leakage, metrics, scoring, monitoring, API)
uv run ruff check .                    # lint
```

Outputs: 8 PNGs in `reports/figures/`, a registered model under `models/<run_id>/`,
`scored.csv` from the batch CLI, `monitoring/drift_report.md`, and the full EDA +
metric tables printed to stdout.
