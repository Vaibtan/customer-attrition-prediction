# ADR 0006 — Lifecycle→serving cutover: split truth, alias-following online, shadow challenger

**Status:** Accepted (design; grilled 2026-07-12) · **Context:** `REVIEW_ISSUES.md` A1/REV-10/REV-14,
extends ADR 0003/0004/0005

## Context

Promotion (ADR 0004) moves the MLflow `@champion` alias, but both serving apps load once from the
file registry and never invalidate — the platform's statistical centerpiece cannot change what
production serves. Making everything follow MLflow would break the batch image's deliberate
self-containment (model baked at build; bare `docker run` serves) and hard-couple serving to the
tracking server; keeping the file registry as truth would make the MLflow registry decorative.

## Decision

**Split source of truth, on purpose.** It mirrors how these paths differ in reality:

- **Batch (`/score`) stays a pinned model:** the baked, build-time artifact; cutover = redeploy.
  Batch jobs pin artifacts.
- **Online (`/score/online`) follows `@champion`:** resolved from MLflow at first request, then
  **lazily re-resolved on the request path when the cached resolution is older than a TTL**
  (`CHAMPION_TTL_SECONDS`; `0` in tests makes cutover deterministic). The artifact reloads only when
  the alias *version* changed. No background thread; single-worker friendly (ISSUES.md §2).
- **Outage policy — serve-last-good:** if re-resolution fails with a model already loaded, keep
  serving it (**stale**, not degraded — see CONTEXT.md); if no model has ever loaded, the app is
  **degraded**. MLflow is therefore a *soft* dependency: the online service (`api-online:8001`, own
  image with no baked model, `serve` profile) has no compose `depends_on` on `mlflow` and recovers
  by itself when tracking appears.
- **Health contract (all apps):** one `/health`, always HTTP 200; the body carries
  `status: ok|degraded|stale` + `model_loaded`, `run_id`, a `reason` on degradation, alias version /
  resolution age for the online app, and `warnings[]` (e.g. recorded-vs-runtime lib-version skew).
  *Any* loader exception degrades — not just `FileNotFoundError`. Scoring 503s only when no model is
  loaded; stale still scores.
- **Shadow challenger (log-only):** if `@challenger` is aliased, the online app also scores the
  **same assembled feature row** with it after the response is sent (FastAPI `BackgroundTasks` —
  zero client latency), appending `{customer_id, as_of, champion/challenger run_ids + scores, ts}`
  to a shadow log and incrementing a Prometheus counter. The response never exposes the challenger;
  no alias → no-op. The log is the future *live-traffic input* to the promotion gate.

## Consequences

- Promotion now really moves production: gate → alias move → online cutover within one TTL, with
  rollback = moving the alias back. The lifecycle story ends in production instead of one step short.
- The asymmetry is the point — do not "fix" batch to follow MLflow: self-containment is what the
  demo image sells, and pinned-batch/alias-online is the realistic pattern.
- Serve-last-good trades bounded staleness for availability; the trade is surfaced (stale state,
  resolution age in `/health`) rather than hidden.
- Rejected: everything-follows-MLflow (kills self-containment, file registry vestigial); champion
  pointer file in the file registry (MLflow decorative, alias semantics maintained twice);
  redeploy-as-cutover documented only (leaves the gap as prose); per-request alias resolution
  (tracking server on the hot path); admin-reload-only (keeps a human between gate and production,
  drags authn into scope now).
