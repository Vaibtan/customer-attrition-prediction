# ADR 0005 — Unified scoring-app scaffold

**Status:** Accepted (Phase 7 cleanup, 2026-07-05) · **Context:** `ISSUES.md` ISS-04; builds on ADR 0003

## Context

Two FastAPI scoring apps exist: `api/serve.py` (`POST /score`, the batch-scoring demo) and
`api/online.py` (`POST /score/online`, the online path of ADR 0003). Each independently reimplemented
the same scaffold — lazy-load-and-cache the model, degrade `/health` to `{degraded, model_loaded:
false}` and fail scoring with `503` on a missing model (`FileNotFoundError`), then attach the
Prometheus metrics middleware — with **divergent** idioms (`serve.py`'s `@lru_cache` keyed by
`run_dir` vs `online.py`'s mutable `state` dict + closure, itself an unlocked check-then-act). The
copies drifted: `online.py` reported `model_run_id` as the configured `run_dir` **path** (or `None`),
not the real registry `run_id`, because `serving.online_model.load_online_model` dropped
`registry.load_run(...).run_id`, and `OnlineModel` had no field to carry it. Two scaffolds that must
be kept in sync by hand is exactly the train/serve-inconsistency-by-drift the platform argues against.

## Decision

Extract one scaffold, `api.scoring_app.scoring_app(...)`, that both apps build on:

- **The scaffold owns** app creation, a **load-once** cached loader, the `FileNotFoundError` →
  `/health` `degraded` / score `503` contract, the `/health` payload
  (`{status, model_loaded, run_id}`), and the metrics middleware (by label).
- **The loader contract is a single knob:** `loader() -> M` where `M` exposes **`.run_id`**.
  `serve.py` passes a `registry.load_run` loader (`LoadedModel` already has `.run_id`); `online.py`
  passes an `OnlineScorer` loader — `OnlineModel` now carries `run_id` (filled by `load_online_model`
  from `loaded.run_id`) and `OnlineScorer` re-exposes it, so the online path reports the **real** id.
- **Everything bespoke lives in a per-app `score_fn(model, req) -> response`:** the batch path builds
  a one-row DataFrame + `validate_schema`; the online path reads Redis (closing over the store),
  raises `404` on a cache miss, and merges the request's static attributes. The scaffold only maps a
  *model-load* failure to `503`; the online **404** stays inside `score_fn`.

Rejected: (a) keeping two hand-synced scaffolds; (b) a *minimal* helper sharing only the boilerplate
while leaving two apps — full unification is clean here **because** no online-only concept leaks into
the scaffold; (c) a generic "resource missing → 404" hook in the scaffold — that would bake the
online cache-miss (a concept `/score` never has) into shared code, a dead branch for the batch path.

## Consequences

- One health/metrics/`503` contract for both endpoints; `/score/online` `/health` and its response
  now carry the real `run_id`, consistent with `/score` (an additive, backward-compatible change).
- The seam is the honesty test for the abstraction: any future divergence must be expressed through
  `score_fn`, keeping the scaffold domain-agnostic — if a change *can't* fit `score_fn`, that is the
  signal the two paths are diverging and should split again.
- `serve.py`'s `@lru_cache(maxsize=8)` multi-run cache collapses to load-once. Equivalent in practice:
  `run_dir` is fixed at app creation, never per request, so the cache only ever held one entry per app.
- Score-level parity (ADR 0003) is untouched — this is a structural refactor of the serving surface,
  not a change to what is served.
