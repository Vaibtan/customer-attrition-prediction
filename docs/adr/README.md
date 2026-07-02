# Architecture Decision Records

One file per decision, numbered `NNNN-short-slug.md`. ADRs capture **architectural /
operational** choices with concrete consequences and actions. The **design and
experimental-protocol** decisions — the pre-registration lock and consequential forks — live in the
decision record `../../PHASE0_LOCK_DECISIONS.md` (D1–D8), and the frozen world in
`../simulator_spec.md`.

- [0001 — Pre-registration lock governance](0001-preregistration-lock-governance.md) — CI guards +
  `main` branch-protection contract.
- [0002 — Real-infra test strategy](0002-real-infra-test-strategy.md) — two tiers; the container
  tier tests against live Redpanda/Redis/MLflow (no mocked infra).
- [0003 — Online serving + parity](0003-online-serving-and-parity.md) — the static+event online
  model, score-level parity, and the O(1) incremental aggregator with a recompute oracle.
- [0004 — Promotion policy](0004-promotion-policy.md) — paired ΔROC-AUC/ΔPR-AUC lower bounds + MDE +
  guardrails; incumbent wins ties.
