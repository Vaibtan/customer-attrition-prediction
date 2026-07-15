# 0008 — Online-store write amplification: pipeline yes, debounce no, dedup TTL no

**Status:** accepted (2026-07-16) · **Context:** ISSUES.md ISS-14, grilled per project convention.

The consumer writes one full feature envelope to Redis per consumed event, and its per-event
dedup markers (`s:{eid}` in Quix state) never expire. Both look like bugs to a fresh reviewer —
that is literally how ISS-14 got filed — so this records that both are deliberate, plus the one
sub-item we did take.

## Decisions

1. **`put_many` batches transport (taken).** `KVBackend` grows `set_many`; the Redis backend
   sends one `MSET` instead of N round-trips. Envelope semantics (`allow_nan=False`, sorted
   keys, strict reader) are unchanged — this is transport batching only. `put_many` currently
   has no production caller (tests seed the parity store with it), but the contract errors
   already promise a "rebuild the online store" path, and that path should not inherit an
   N-round-trip footgun.

2. **Write-per-event stays; the debounce is rejected.** Only the final vector is ever read, so
   batching writes per customer looks free. It is not: today every event's Redis write lands
   before its offset can be committed, so at-least-once redelivery always rewrites — Redis can
   lag but never permanently miss the tail. A debounce needs a dirty-customer set flushed
   before every offset commit; getting that ordering wrong leaves Redis silently diverged from
   the offline PIT vector — a parity break in the exact code path whose byte-parity is the
   platform's brand. The measured ~3.2k events/s was achieved WITH per-event writes; at demo
   scale the saved SETs buy nothing. The "freshness semantics" worry in the original finding is
   a red herring: the reader contract is `as_of` equality (ADR 0007), not recency, and a
   mid-replay read returns a partial-prefix vector under either scheme.

3. **Dedup markers stay immortal; the TTL/LRU bound is rejected.** The frozen world produces no
   new data by design, so marker state is bounded by the finite event log — "unbounded growth"
   assumes a live world this platform deliberately does not have. The TTL side is genuinely
   dangerous: a marker evicted before a redelivery arrives makes the incremental aggregator
   fold the event twice, permanently and silently poisoning parity. Quix state has no native
   TTL, so the bound would be hand-rolled eviction inside the idempotency mechanism itself.
   **If the world ever unfreezes**, the documented path is a TTL sized to the broker's
   redelivery window, accepted explicitly as a correctness-vs-memory trade — not built now.
