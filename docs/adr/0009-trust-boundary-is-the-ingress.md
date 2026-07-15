# 0009 — The trust boundary is the ingress; the app layer ships no authn

**Status:** accepted (2026-07-16) · **Context:** REVIEW_ISSUES.md architecture item 8.

The scoring APIs (`/score`, `/score/online`) are unauthenticated by design, not oversight. The
trust boundary of this platform is the ingress: in production these services would sit behind a
gateway / service mesh that terminates authn (OIDC or mTLS) and owns rate limiting, and the app
trusts the network behind it. Duplicating that at the app layer in this demo would mean a static
API key in a compose file — security theater that protects nothing real and normalizes a weak
pattern — or a key enforced only when configured, which is fail-open and worse than an honest
boundary statement.

**Consequences:** anyone deploying this beyond a private compose network must put a real
ingress in front (gateway-terminated authn + rate limiting); nothing in-app will stop traffic.
The same posture governs the rest of the stack: secrets live in an untracked `.env`
(REVIEW_ISSUES.md REV-16), CI runs least-privilege, and `/metrics` stays unauthenticated for the
scraper on the internal network.
