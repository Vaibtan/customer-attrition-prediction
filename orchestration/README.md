# orchestration/

Dagster code location (`dagster dev -m orchestration.definitions`; the compose `orchestration`
profile serves the UI on host port **3001** — Grafana owns 3000).

- `assets.py` — the Phase-2 vertical slice: generate → offline PIT → online → `parity_report`,
  which **fails the run** if online/offline diverge (parity as a materialisable gate).
- `timeline.py` — the static-partitioned scoring timeline (`pit_snapshot` over four historical
  t0s) + the drift-gated conditional-retrain asset.
- `definitions.py` — jobs + schedules. Two deliberate shapes (REVIEW_ISSUES.md REV-03):
  - `timeline_backfill` is **backfill-only** (UI/CLI): a static, historical partition set over a
    frozen world has no "next" partition, so a cron over it is incoherent — and a bare
    `ScheduleDefinition` tick can't launch a partitioned job anyway.
  - `weekly_retrain` (Mondays 06:00) runs the **unpartitioned** drift-gated retrain branch —
    detect drift, retrain + gate only if it fires. A tick-evaluation test pins that the schedule
    actually launches (`tests/test_orchestration_timeline.py`).
