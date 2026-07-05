"""Cohort frame builder -- the ``(customer_id, t0)`` input to the offline PIT query."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def make_cohort(ids: Iterable, t0) -> pd.DataFrame:
    """Build the ``(customer_id, t0)`` cohort frame: every id queried as of the same instant ``t0``.

    The single definition of the cohort-frame shape that
    :func:`churn.featurestore.offline.compute_pit_features` consumes -- previously re-hardcoded at a
    dozen call sites. ``ids`` may be any iterable (materialised once so a generator is safe).
    """
    ids = list(ids)
    return pd.DataFrame({"customer_id": ids, "t0": [t0] * len(ids)})
