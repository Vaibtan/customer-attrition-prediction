# AI Analyst Exercise — Customer Churn Prediction

> **Time estimate:** 2–3 hours  
> **Submission:** Share a GitHub repo with your code, outputs, and a brief write-up.

## Implementation Status

This repo contains the completed solution, not just the starter script. The
implementation keeps the pipeline leak-free, does **not** apply target-imbalance
handling (the label is balanced at 48.4% churn), cleans dirty rows inside the
sklearn pipeline, and frames the result honestly around the measured weak signal
(roughly 0.62 ROC-AUC).

Run the core workflow:

```bash
uv sync --all-groups
uv run python churn_prediction.py
uv run pytest -q
```

Batch scoring, monitoring, and the optional FastAPI demo:

```bash
uv run python -m churn.scoring --in data/customer_data.csv --out scored.csv
uv run python -m churn.monitoring --reference data/customer_data.csv --current data/customer_data.csv
uv sync --extra serve --all-groups
uv run uvicorn api.serve:app --reload
```

The Docker image is **self-contained** — it trains and registers a model at build
time, so a bare `docker run` serves `/score` with no mounted volume:

```bash
docker build -t churn-scoring .
docker run --rm -p 8000:8000 churn-scoring
# then: curl localhost:8000/health   →  {"status":"ok","model_loaded":true,...}
```

The main run writes PNG figures to `reports/figures/` and a registered model to
`models/<run_id>/`. `WRITEUP.md` contains the reviewer-facing explanation and
`IMPLEMENTATION_PLAN.md` maps implementation choices to the locked D1-D6 scope.

---

## Background

An e-commerce company is experiencing higher-than-expected customer attrition. The growth team wants to proactively identify customers who are likely to churn so that retention campaigns can be targeted effectively.

You have been given a dataset of **1,600 customer records**, each labelled with whether the customer churned within the following 90 days. Your task is to explore the data, build a churn prediction model, and surface insights that could guide retention strategy.

---

## Repository Structure

```
├── data/
│   └── customer_data.csv ← Dataset (do not modify)
├── src/churn/           ← Leak-free training, cleaning, scoring, registry, monitoring
├── api/
│   └── serve.py         ← FastAPI /score demo using the registered pipeline
├── tests/               ← Contract, leakage, metrics, scoring, monitoring tests
├── reports/figures/     ← Generated PNG deliverables
├── models/              ← Generated local model registry entries
├── churn_prediction.py  ← End-to-end entry point
├── WRITEUP.md           ← Decisions, findings, and interpretation
├── IMPLEMENTATION_PLAN.md
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md            ← This file
```

---

## Dataset Description

**File:** `data/customer_data.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer identifier |
| `region` | string | Customer's geographic region |
| `device_type` | string | Primary device used (Mobile, Desktop, Tablet) |
| `subscription_plan` | string | Current plan (Free, Basic, Premium, Enterprise) |
| `account_age_days` | int | Days since account creation |
| `monthly_spend` | float | Average monthly spend over last 6 months (USD) |
| `num_orders_last_90d` | int | Number of orders placed in last 90 days |
| `avg_order_value` | float | Average value per order (USD) |
| `support_tickets_raised` | int | Support tickets opened in last 6 months |
| `days_since_last_login` | int | Days since the customer last logged in |
| `pages_per_session` | float | Average pages visited per session |
| `churned` | int | Target — 1 if customer churned, 0 otherwise |

> The dataset has class imbalance across subscription plans, missing values in some behavioural columns, a small proportion of outlier spend values, and a few records with logically impossible field values.

---

## Tasks

### Part 1 — Exploratory Data Analysis
1. Summarise the dataset: shape, types, missing values, and descriptive statistics.
2. Analyse the overall churn rate and churn rate broken down by `subscription_plan` and `region`.
3. Compare behavioural features (e.g. `days_since_last_login`, `monthly_spend`, `support_tickets_raised`) between churned and retained customers using appropriate plots.
4. Comment on any class imbalance you observe and how you plan to address it.

### Part 2 — Data Cleaning & Preprocessing
1. Identify and correct impossible values across all columns (think about valid real-world ranges).
2. Detect and handle outliers — document your method and rationale.
3. Impute missing values with a justified strategy for each column.
4. Encode categorical variables and scale features appropriately.
5. Engineer at least one new feature that you think could be predictive.

### Part 3 — Review & Improve the Starter Code
Open `churn_prediction.py` and the `src/churn/` package. They supersede the
starter pipeline with a leak-free implementation; the full defect catalog and
fix rationale are documented in `WRITEUP.md`.

Read through the code carefully. Some sections may produce misleading outputs, use incorrect metric implementations, or have subtle logical issues. Identify what needs fixing, apply your corrections with brief explanations, and extend the code to complete all parts of the exercise.

### Part 4 — Model Building & Evaluation
1. Train at least **two classification models**.
2. Report **Precision, Recall, F1-score, and ROC-AUC** for each model on the test set.
3. Plot the **ROC curve** for both models on the same axes.
4. Plot and interpret the **confusion matrix** for your best model.
5. Discuss which metric matters most in a business churn context and why.

### Part 5 — Business Interpretation *(encouraged)*
- Which customer profile is most at risk of churning?
- What retention action would you recommend for the highest-risk segment?
- How would you measure the real-world impact of deploying this model?

---

## Environment Setup

```bash
uv sync --all-groups
```

For the FastAPI demo, install the serving extra:

```bash
uv sync --extra serve --all-groups
```
---

## Submission Checklist

- [x] `churn_prediction.py` — complete and runnable end-to-end
- [x] All plots saved as `.png` under `reports/figures/`
- [x] `WRITEUP.md` covering decisions and findings
- [x] README Parts 1-5 implemented in `src/churn/`, `tests/`, and `churn_prediction.py`
- [x] Batch scoring CLI emits `customer_id`, `churn_probability`, `risk_tier`, and reason codes
- [x] Local registry, monitoring report, FastAPI demo, Dockerfile, and CI workflow included
