"""The three named baselines + positive/negative controls (§4.4/§4.5) -- the instrument's readout.

Emits the three distinctly-named baselines (never substituted for one another):
``real_static_reference_auc`` (static features, **real** labels, anchor -- reported alone),
``synthetic_static_auc`` (static, simulated labels), and ``synthetic_static_plus_event_auc``
(static + event, simulated labels -- the positive-control headline). The positive control is the
paired-bootstrap ΔROC-AUC / ΔPR-AUC lower bound; the negative controls (label-shuffle, null-stream)
must show no lift. Customer-level resampling (D4) keeps the bootstrap unit = customer.

Part of the ANALYSIS set hashed at Stage 2 (D1/D2).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.instrument import model as M


def paired_diff_ci(
    y: NDArray,
    proba_a: NDArray,
    proba_b: NDArray,
    metric: Callable[[NDArray, NDArray], float],
    groups: NDArray | None = None,
    n_rounds: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile paired bootstrap CI for ``metric(a) - metric(b)`` (customer-level if ``groups``).

    Resamples the **bootstrap unit** with replacement -- unique ``groups`` (customers) when given,
    else rows -- so the CI respects the group structure (D4). Single-class resamples are skipped.
    """
    y = np.asarray(y)
    a = np.asarray(proba_a, dtype="float64")
    b = np.asarray(proba_b, dtype="float64")
    rng = np.random.default_rng(seed)

    if groups is None:
        n = len(y)
        draws = (rng.integers(0, n, size=n) for _ in range(n_rounds))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
        draws = (
            np.concatenate([rows_by_group[g] for g in rng.choice(uniq, size=len(uniq))])
            for _ in range(n_rounds)
        )

    diffs = []
    for idx in draws:
        if np.unique(y[idx]).size < 2:
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": float(np.quantile(diffs, alpha / 2)),
        "diff_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "n_rounds": int(diffs.size),
    }


def compute_baselines(dataset, pit_features, seed: int = 42) -> dict:
    """The three named baselines + the raw OOF predictions (for the paired positive-control CI)."""
    design = M.assemble_design(dataset, pit_features)
    real = M.assemble_design(dataset, pit_features, use_real_label=True)

    p_static = M.group_oof_proba("static", design.X, design.y, design.groups, seed=seed)
    p_event = M.group_oof_proba("static_event", design.X, design.y, design.groups, seed=seed)
    p_real = M.group_oof_proba("static", real.X, real.y, real.groups, seed=seed)
    oracle = dataset.customers.loc[dataset.customers["is_anchor"], "oracle_score"].to_numpy()

    return {
        "n": int(len(design.y)),
        "base_rate": float(design.y.mean()),
        "real_static_reference_auc": M.auc(real.y, p_real),
        "real_static_reference_pr": M.pr_auc(real.y, p_real),
        "synthetic_static_auc": M.auc(design.y, p_static),
        "synthetic_static_pr": M.pr_auc(design.y, p_static),
        "synthetic_static_plus_event_auc": M.auc(design.y, p_event),
        "synthetic_static_plus_event_pr": M.pr_auc(design.y, p_event),
        "oracle_auc": M.auc(design.y, oracle) if len(oracle) == len(design.y) else float("nan"),
        "_y": design.y,
        "_groups": design.groups,
        "_p_static": p_static,
        "_p_event": p_event,
    }


def positive_control(
    dataset, pit_features, seed: int = 42, bootstrap_seed: int = 42, n_rounds: int = 1000
) -> dict:
    """Paired ΔROC-AUC / ΔPR-AUC bootstrap of (static+event) - static, customer-level (D4)."""
    base = compute_baselines(dataset, pit_features, seed=seed)
    y, groups, p_static, p_event = (
        base["_y"],
        base["_groups"],
        base["_p_static"],
        base["_p_event"],
    )
    return {
        "delta_roc": paired_diff_ci(
            y,
            p_event,
            p_static,
            roc_auc_score,
            groups=groups,
            n_rounds=n_rounds,
            seed=bootstrap_seed,
        ),
        "delta_pr": paired_diff_ci(
            y,
            p_event,
            p_static,
            average_precision_score,
            groups=groups,
            n_rounds=n_rounds,
            seed=bootstrap_seed + 1,
        ),
    }
