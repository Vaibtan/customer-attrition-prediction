"""Drift simulation (ENHANCEMENT_PLAN.md Sec 4): the three drift KINDS x three temporal PROFILES.

- **kinds** -- covariate (feature distribution moves), prior (label base rate moves), concept
  (the feature -> label relationship changes; the one CBPE is provably blind to).
- **profiles** -- ``sudden`` / ``gradual`` / ``recurring`` return the drift magnitude in [0, 1] at
  step ``t`` of ``n``, so a backtest can drive any kind through any temporal shape.

Injectors take a base frame + a magnitude in [0, 1] and return a drifted copy, so a replay harness
composes ``profile(t) -> magnitude -> injector`` to build a drifting timeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- temporal magnitude profiles: fraction of full drift at step t of n (values in [0, 1]) ------


def sudden(t: int, n: int, change_point: float = 0.5) -> float:
    """Step change: 0 before the change point, full magnitude after."""
    return 0.0 if t < change_point * n else 1.0


def gradual(t: int, n: int) -> float:
    """Linear ramp from 0 at t=0 to 1 at t=n-1."""
    return 0.0 if n <= 1 else float(t) / float(n - 1)


def recurring(t: int, n: int, cycles: float = 2.0) -> float:
    """Seasonal oscillation in [0, 1] (raised cosine), returning to 0 each cycle."""
    return float((1.0 - np.cos(2.0 * np.pi * cycles * t / max(n, 1))) / 2.0)


# --- drift injectors (magnitude in [0, 1]) ------------------------------------------------------


def covariate_shift(df: pd.DataFrame, column: str, magnitude: float, delta: float) -> pd.DataFrame:
    """Shift a numeric feature's location by ``magnitude * delta`` (covariate drift)."""
    out = df.copy()
    out[column] = pd.to_numeric(out[column], errors="coerce") + magnitude * delta
    return out


def categorical_shift(
    df: pd.DataFrame, column: str, magnitude: float, to_level: object, seed: int = 0
) -> pd.DataFrame:
    """Move a ``magnitude`` fraction of a categorical column's mass onto ``to_level``."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    flip = rng.random(len(out)) < magnitude
    out.loc[flip, column] = to_level
    return out


def prior_shift(df: pd.DataFrame, target: str, magnitude: float, seed: int = 0) -> pd.DataFrame:
    """Raise the positive base rate by relabelling a ``magnitude`` fraction of negatives to 1."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    negatives = out.index[out[target] == 0]
    n_flip = int(round(magnitude * len(negatives)))
    if n_flip:
        out.loc[rng.choice(negatives, size=n_flip, replace=False), target] = 1
    return out


def concept_shift(
    df: pd.DataFrame,
    feature: str,
    target: str,
    magnitude: float,
    threshold: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Rewire ``feature -> target`` for a ``magnitude`` fraction of rows (concept drift).

    The relationship, not the marginals, changes: for the affected rows the label is set from a
    thresholded ``feature`` (high feature -> churn), so a model fit on the old relationship degrades
    while covariate detectors on ``feature`` see nothing (its marginal is unchanged).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    values = pd.to_numeric(out[feature], errors="coerce")
    cut = values.median() if threshold is None else threshold
    affected = rng.random(len(out)) < magnitude
    out.loc[affected, target] = (values[affected] > cut).astype(int)
    return out
