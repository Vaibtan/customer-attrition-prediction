# ADR 0003 — Online serving model + score-level parity

**Status:** Accepted (Phase 3, 2026-07-02) · **Context:** `PHASE0_LOCK_DECISIONS.md` D8.1, D8.3

## Context

The scoped online path (ENHANCEMENT_PLAN.md Sec 3) needs a model that consumes the online
event-feature vector, and the online features must equal the offline PIT features so on-demand
scores match the batch decision. Two sub-problems: which model serves online, and how the streaming
aggregation stays byte-identical to the DuckDB PIT query at production event volumes.

## Decision

- **Online model:** the instrument's `static_event` pipeline (`serving.online_model`), fit on the
  synthetic population's simulated labels with campaign-cost tier cutpoints. `/score/online`
  (`api/online.py`) reads the event vector from Redis by `customer_id` and takes static attributes
  from the request. Because online event features == offline PIT features, the served score **equals
  the offline batch score** for the same `(customer, t0)`. **Honesty scope:** a synthetic-domain
  systems demo of the online path, never a real-world performance claim (Sec 1).
- **One reference reducer + a fast path:** `streaming.aggregate.feature_vector` (recompute) is the
  obvious, correct reducer and the **test oracle**. The deployed consumer uses
  `CustomerAggregator` — an **O(1)-per-event** incremental aggregator (window membership is static
  for a fixed `t0`) with fixed-size state and per-key dedup. The parity suite pins **SQL offline ==
  recompute == incremental** under every adversarial fixture, so the fast path inherits correctness.

## Consequences

- Train/serve consistency is proven at BOTH the feature and the score level, live (ADR 0002).
- Throughput is ~3k events/s in the local integration test rather than the ~200/s of a naive
  recompute-per-event consumer, without weakening the parity guarantee.
- A second, independent online implementation in Quix's window DSL was rejected: it could silently
  diverge from the SQL and is harder to prove byte-identical.
- Static attributes come from the request (the caller already holds the profile), so no online
  profile store is needed for the demo's scope.
