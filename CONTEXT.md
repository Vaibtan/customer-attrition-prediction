# Churn Platform

Ubiquitous language for the churn-prediction MLOps platform (branch `enhancement`): an
honest-diagnosis modeling core wrapped in a production platform — streaming parity features,
lifecycle with statistical promotion gates, drift monitoring, and observability — over a frozen
synthetic world.

## Language

### World & data

**Frozen world**:
The simulated customer universe, pre-registration-locked at a fixed anchor instant; it produces no
new data after that instant by design (D5).
_Avoid_: live data, production data

**as_of**:
The scoring instant a feature vector was computed for. In the frozen world this is a fixed t0, so
online freshness means *as_of equality* with the scorer's configured instant — never a wall-clock
TTL.
_Avoid_: timestamp, snapshot time (ambiguous)

**Feature vector envelope**:
The versioned online payload `{schema, as_of, features}` written to the online store. `schema` is
derived from the canonical feature-column spec (changes by construction when the spec changes, no
manual bump).
_Avoid_: raw feature dict, blob

### Models & lifecycle

**Champion**:
The model production currently serves, designated by the MLflow `@champion` alias.

**Challenger**:
A gate-candidate model, designated `@challenger`; it may shadow-score live traffic but never
answers a caller.

**Shadow scorer**:
Log-only scoring of the challenger on the *same* assembled inputs as the champion, recorded for
future promotion evidence; invisible in responses and free when no challenger is aliased.
_Avoid_: A/B test, canary (those route real traffic; shadow routes none)

**Pinned model**:
A model resolved once from an immutable artifact and never swapped at runtime. The batch app is
pinned; the online app is alias-following.
_Avoid_: baked model (packaging detail, not the contract)

**Cutover**:
The moment serving starts answering with a newly promoted champion. For the online app this is the
alias re-resolution taking effect; for the batch app it is a redeploy.

**Degraded**:
The serving state in which no model is loaded for *any* reason; health reports it with a reason and
scoring refuses (503).
_Avoid_: down, unhealthy (the process is fine)

**Stale**:
The serving state in which the last-good champion still answers while alias re-resolution is
failing. Stale still scores; health exposes the age.
_Avoid_: degraded (stale serves; degraded refuses)

### Monitoring & time

**Label horizon**:
The delay (in replay steps) between scoring a window and its labels becoming available. Stylized in
the backtest and said so; label-free signals (drift, CBPE) act inside the horizon, realized metrics
only after it.
_Avoid_: label lag (use horizon consistently)

**Backfill**:
Materializing a static, historical partition set on demand. The scoring timeline is backfill-only —
a recurring cron over a finite past is theater; the weekly schedule belongs to the retrain branch.
_Avoid_: scheduled run (that's the retrain branch)

### Configuration

**Runtime settings**:
Env-tunable infrastructure knobs (URLs, topics, TTLs), owned by one typed settings object.
_Avoid_: config (reserved for policy)

**Policy**:
Frozen ML decision parameters (thresholds, MDEs, tolerances) in `config.py`; deliberately *not*
env-overridable — tuned once and frozen with the policy they implement.
_Avoid_: settings, tunables
