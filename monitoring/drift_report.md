# Churn Monitoring Drift Report

- Reference rows: 1,600
- Current rows: 1,600
- Model run: 20260619T182613Z_logistic_regression

## Data Quality

| column | missing | invalid_range |
| --- | --- | --- |
| region | 0 | 0 |
| device_type | 0 | 0 |
| subscription_plan | 0 | 0 |
| account_age_days | 0 | 8 |
| monthly_spend | 80 | 0 |
| num_orders_last_90d | 0 | 0 |
| avg_order_value | 80 | 0 |
| support_tickets_raised | 0 | 0 |
| days_since_last_login | 0 | 10 |
| pages_per_session | 80 | 0 |

## Feature Drift

| feature | psi | ks |
| --- | --- | --- |
| region | 0.000 |  |
| device_type | 0.000 |  |
| subscription_plan | 0.000 |  |
| account_age_days | 0.000 | 0.000 |
| monthly_spend | 0.000 | 0.000 |
| num_orders_last_90d | 0.000 | 0.000 |
| avg_order_value | 0.000 | 0.000 |
| support_tickets_raised | 0.000 | 0.000 |
| days_since_last_login | 0.000 | 0.000 |
| pages_per_session | 0.000 | 0.000 |

## Prediction Drift

- Score PSI: 0.000
- Reference mean score: 0.485
- Current mean score: 0.485

## Alert Guide

- PSI < 0.10: stable.
- PSI 0.10-0.20: watch.
- PSI > 0.20: investigate before trusting campaign decisions.
