# ADR 0002 — Real-infra test strategy (two tiers, no mocked infra)

**Status:** Accepted (Phase 2, 2026-07-02) · **Context:** `PHASE0_LOCK_DECISIONS.md` D8.2

## Context

The platform's headline guarantee is train/serve consistency across a broker + online store. A test
that mocks Redpanda/Redis/MLflow can only prove the mock, not the deployment. But the streaming
clients (`confluent-kafka`, `quixstreams`) are Linux-container-only and the dev box is Windows.

## Decision

Two test tiers, split by a pytest marker:

- **Host tier (fast, infra-free):** pure logic + in-process online/offline parity, run natively.
  A `DictBackend` fake stands in for Redis for store round-trips only. Default `pytest` run; CI's
  `test` job runs `-m "not integration"`.
- **Container tier (canonical):** `@pytest.mark.integration` tests run inside a Dockerised
  `test-runner` (`infra/Dockerfile.test`, all extras) networked to **live** Redpanda + Redis +
  MLflow via `docker compose --profile test`. The whole repo is bind-mounted read-only so source
  edits need no rebuild; connection info comes from env (`REDPANDA_BROKER`, `REDIS_URL`,
  `MLFLOW_TRACKING_URI`) and host runs skip cleanly when it is absent.

Run locally: `docker compose --profile test run --rm test-runner`. CI's `integration` job runs it.

## Consequences

- The parity/serving/lifecycle guarantees are proven through the ACTUAL deployment path (produce ->
  Quix consumer -> Redis -> read; log -> MLflow registry -> alias), not a stand-in.
- The single online reference reducer (`streaming.aggregate`) is imported by BOTH the parity test and
  the deployed consumer, so parity is a real cross-check (SQL vs Python), never a tautology (D8.1).
- The host tier keeps the inner loop fast; the container tier is the gate before "green".
- Cost: a slower CI `integration` job and a one-time image build. Acceptable for the guarantee.
