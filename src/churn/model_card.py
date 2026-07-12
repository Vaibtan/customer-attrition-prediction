"""Render a per-run MODEL_CARD.md from registry metadata (REVIEW_ISSUES.md architecture item 7).

The metadata ``save_run`` writes already carries most of a model card (provenance, the selection
honesty statistics, the decision policy, the feature lists); this module renders it as the named
artifact reviewers look for. Rendering is pure (``dict -> str``) so ``registry`` can import it
without a cycle, and defensive (every key optional) so minimal runs still get a card.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_CARD_FILE = "MODEL_CARD.md"


def _f(x, spec: str = ".3f") -> str:
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "n/a"


def _kv_table(mapping: dict, key_header: str, value_header: str) -> list[str]:
    lines = [f"| {key_header} | {value_header} |", "| --- | --- |"]
    lines += [f"| {k} | {_f(v)} |" for k, v in mapping.items()]
    return lines


def _cell(v) -> str:
    return v if isinstance(v, str) else _f(v)


def _rows_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(none recorded)"]
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c, "")) for c in cols) + " |")
    return lines


def render(meta: dict) -> str:
    """Metadata dict (the ``metadata.json`` content) -> the model card markdown."""
    cv = meta.get("cv_results") or {}
    sel = meta.get("selection") or {}
    ci = sel.get("winner_auc_ci") or {}
    diff = sel.get("winner_vs_runner_up_auc_diff_ci") or {}
    econ = meta.get("cost_params") or {}
    cuts = meta.get("tier_cutpoints") or {}
    versions = meta.get("versions") or {}

    lines: list[str] = [
        f"# Model card -- {meta.get('model_name', 'model')} ({meta.get('run_id', 'unknown run')})",
        "",
        "## Intended use",
        "",
        "Rank customers of the SYNTHETIC churn domain for a retention campaign: batch scoring is",
        "the production decision path (risk tiers + reason codes), the online API is the",
        "low-latency mirror of the same artifact. This is a systems demo on a simulated world --",
        "no number below is a real-world performance claim.",
        "",
        "## Provenance",
        "",
        f"- created_at: {meta.get('created_at', 'n/a')}",
        f"- git_sha: {meta.get('git_sha', 'n/a')}",
        f"- data_sha256: {meta.get('data_sha256', 'n/a')}",
        f"- seed: {meta.get('seed', 'n/a')}",
        f"- versions: {json.dumps(versions) if versions else 'n/a'}",
        "",
        "## Selection (bake-off, honesty statistics)",
        "",
    ]
    if cv:
        lines += [
            "| model | CV ROC-AUC mean | std |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| {name} | {_f(r.get('cv_auc_mean'))} | {_f(r.get('cv_auc_std'))} |"
            for name, r in cv.items()
        ]
    else:
        lines.append("(no bake-off recorded)")
    lines += [
        "",
        f"Winner picked as the SIMPLEST model within tolerance {_f(sel.get('tolerance'))} of the",
        f"best CV AUC (runner-up: {sel.get('runner_up', 'n/a')}). Hold-out winner AUC 95% CI:",
        f"[{_f(ci.get('auc_lo'))}, {_f(ci.get('auc_hi'))}]; paired winner-vs-runner-up delta CI:",
        f"[{_f(diff.get('diff_lo'))}, {_f(diff.get('diff_hi'))}] -- a CI straddling 0 means the",
        "bake-off was a tie and simplicity broke it.",
        "",
        "## Hold-out performance",
        "",
        "At the business threshold:",
        "",
        *_kv_table(meta.get("metrics_test_business_threshold") or {}, "metric", "value"),
        "",
        "At the conventional 0.5 threshold:",
        "",
        *_kv_table(meta.get("metrics_test_half_threshold") or {}, "metric", "value"),
        "",
        "## Decision policy",
        "",
        f"- campaign threshold t* = {_f(meta.get('threshold'))} (chosen on train OOF expected",
        "  value, never on the hold-out)",
        f"- tier cutpoints: t_mid = {_f(cuts.get('t_mid'))}, t_star = {_f(cuts.get('t_star'))}",
        f"- economics: cost/contact = {_f(econ.get('cost_per_contact'), '.2f')},"
        f" value/retained = {_f(econ.get('value_per_retained'), '.2f')},"
        f" uplift = {_f(econ.get('campaign_uplift'))}",
        "- expected value at t* (train OOF, total): "
        + _f(meta.get("expected_value_at_threshold"), ".1f"),
        "",
        "EV sensitivity across economic scenarios:",
        "",
        *_rows_table(meta.get("ev_sensitivity") or []),
        "",
        "## Features",
        "",
        f"- numeric: {', '.join(meta.get('numeric_features') or []) or 'n/a'}",
        f"- categorical: {', '.join(meta.get('categorical_features') or []) or 'n/a'}",
        "",
        "## Limitations (read before trusting a number)",
        "",
        "- The domain is a frozen synthetic world; the signal ceiling is set by the simulator,",
        "  and results do not transfer to any real customer base.",
        "- Probabilities are sigmoid-calibrated on the train split; calibration can decay in",
        "  production before discrimination does.",
        "- The label arrives ~90 days late: live monitoring leans on CBPE, which is blind to",
        "  concept drift by construction (its band is sampling variance only -- a band miss can",
        "  also be plain calibration failure).",
        "- Promotion is gated on paired bootstrap lower bounds vs an MDE with",
        "  incumbent-wins-ties; a better-by-noise challenger is deliberately NOT promoted.",
        "- Library version skew between artifact and runtime is checked at load and surfaced as",
        "  warnings, not errors.",
        "",
    ]
    return "\n".join(lines)


def write_card(run_dir: Path | str) -> Path:
    """Render (or re-render) the card for an existing run directory from its metadata.json."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    out = run_dir / MODEL_CARD_FILE
    out.write_text(render(meta))
    return out


if __name__ == "__main__":
    import argparse

    from . import registry

    parser = argparse.ArgumentParser(description="Render MODEL_CARD.md for a saved run.")
    parser.add_argument("--run-dir", default=None, help="Run directory (default: latest).")
    args = parser.parse_args()
    target = args.run_dir if args.run_dir is not None else registry.latest_run_dir()
    print(f"Wrote {write_card(target)}")
