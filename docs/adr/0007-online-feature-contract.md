# ADR 0007 — Online feature contract: versioned envelope, strict reader

**Status:** Accepted (design; grilled 2026-07-12) · **Context:** `REVIEW_ISSUES.md` REV-11/A3,
extends ADR 0003

## Context

The online store held a bare flat JSON dict; the reader median-imputed anything missing and trusted
everything present. A stale or partially-written vector therefore produced a confidently wrong
"parity" score — the worst possible failure for a platform whose headline guarantee is byte-parity.
Schema evolution and freshness were invisible.

## Decision

- **Envelope:** the online payload becomes `{schema, as_of, features}`.
- **Derived version, not a manual bump:** `schema` is a short hash derived from the canonical
  feature-column spec, so it changes *by construction* when the spec changes. A hand-bumped integer
  was rejected: it reintroduces the forget-to-bump failure mode (the ISS-09 lesson) on the most
  parity-critical seam.
- **The consumer stamps at write**, and serializes with `json.dumps(..., allow_nan=False)` — a NaN
  feature fails loudly at the producer instead of poisoning the store (bounds the REV-04 class of
  latent divergence).
- **Strict reader — parity scores are real or refused:** missing key stays 404; schema mismatch,
  `as_of` mismatch, or an incomplete feature set → **409 with the reason. Never impute.**
- **Freshness is `as_of` equality, not a TTL:** in the frozen world (D5) the scoring instant is a
  fixed t0, so "fresh" means "computed for the instant this scorer is configured for".

## Consequences

- A feature-spec change now surfaces as an explicit 409 (stale blobs are detectable and re-buildable
  by re-running the consumer) instead of silently scoring against the wrong schema.
- Availability is deliberately sacrificed for truthfulness on this endpoint: a lenient
  impute-and-flag reader was rejected because a flagged-but-fabricated "parity score" contradicts
  the guarantee the platform exists to demonstrate.
- Existing Redis blobs are invalidated once (no migration; the consumer repopulates); parity
  fixtures and the serving integration test update to the envelope.
