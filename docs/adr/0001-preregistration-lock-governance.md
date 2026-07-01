# ADR 0001 — Pre-registration lock governance

**Status:** Accepted (Phase 0, 2026-07-01) · **Context:** `PHASE0_LOCK_DECISIONS.md` D2

## Context

The Stage-1 pre-registration lock (`simulator.lock.json`) is enforced by two CI guards and one
**repository setting**. The procedural guard (Layer 2) and the "conspicuous, reviewed, results-free
re-lock" stopping rule (D3) only hold **behind branch protection** — otherwise a direct push could
bypass review and land a frozen-set change alongside results.

## Decision

Enforce, in code + CI (already in `.github/workflows/ci.yml`):

- **Layer 1 — consistency** (`lock` job, push + PR): `python -m churn.simulator.lock` recomputes
  every FROZEN-file hash (line-ending-normalized) + checks the committed beacon recipe against the
  code; non-zero exit on drift.
- **Layer 2 — same-commit separation** (`lock-separation` job, PR only, `fetch-depth: 0`): using the
  `git merge-base`..HEAD diff, fail if a single change set touches **both** the FROZEN set and the
  RESULTS set.
  - FROZEN (Stage 1): `docs/simulator_spec.md`, `simulator.params.json`, `simulator.lock.json`, the
    world-construction + **enforcement** code (`kernels.py`, `params.py`, `beacon.py`, `lock.py`) and
    `.github/workflows/ci.yml` itself — so a results PR cannot weaken the checker/guard while staying
    Layer-2-clean. **Stage 2 adds** `analysis_spec.json` + the ANALYSIS code modules.
  - RESULTS: `reports/instrument_validation/**` (appear at Stage 2).

Require, as a **GitHub repository setting on `main`** (manual — cannot be set from the repo):

1. **Protect `main`**: require a pull request before merging; **no direct pushes**.
2. **Require review**: at least one approving review on every PR, especially any **re-lock** PR
   (one that regenerates `simulator.lock.json`).
3. **Require status checks to pass**: `test`, `lock`, and `lock-separation` (and `docker`).
4. **Do not allow bypassing** the above (no force-push, no admin override for merges).

## Consequences

- Changing the experiment requires a deliberate, conspicuous, reviewed, **results-free** re-lock
  commit — which, via the beacon round rule (D3), also yields a fresh, still-unpredictable seed.
- A solo author cannot silently roll the dice twice or co-modify the bar and the outcome.
- **Open action (manual):** enable the `main` branch-protection rules above in GitHub repo settings.
  Until then, Layers 1–2 run but the procedural guarantee is advisory only. Consider a CODEOWNERS
  entry on the FROZEN + enforcement paths for required reviewer sign-off.
- **Stage-2 reproducibility prerequisite:** the tamper-evident *regeneration* of confirmatory
  predictions + floors (D2/D4) needs bit-reproducible CI. The `uv` version is now pinned; before
  Stage 2 also pin the runner image by digest and the Python patch version (CI currently floats
  `ubuntu-latest` and Python `3.11`), so a regeneration cannot drift across reruns.
