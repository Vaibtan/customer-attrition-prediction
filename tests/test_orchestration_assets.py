"""Dagster slice DAG: materialises end-to-end and gates online/offline parity.

Pure in-process orchestration (no broker/store), so this runs anywhere Dagster is installed -- a
fast gate that the whole generate -> PIT -> online -> parity graph wires up and the parity asset
holds. The live-infra transport is covered by ``tests/test_streaming_integration.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dagster")

from dagster import Definitions, materialize  # noqa: E402

from orchestration import assets as A  # noqa: E402
from orchestration import definitions as D  # noqa: E402

_SLICE = [A.synthetic_dataset, A.offline_pit_features, A.online_features, A.parity_report]


def _materialize(n_customers: int):
    return materialize(_SLICE, resources={"slice_config": A.SliceConfig(n_customers=n_customers)})


def test_slice_dag_materializes_and_parity_holds():
    result = _materialize(n_customers=20)
    assert result.success

    mats = result.asset_materializations_for_node("parity_report")
    assert len(mats) == 1
    meta = mats[0].metadata
    assert meta["parity_ok"].value is True
    assert meta["n_customers"].value == 20
    assert meta["max_abs_diff"].value < 1e-9


def test_definitions_load_all_slice_assets():
    assert isinstance(D.defs, Definitions)
    keys = {k.to_user_string() for k in D.defs.resolve_all_asset_keys()}
    for name in ("synthetic_dataset", "offline_pit_features", "online_features", "parity_report"):
        assert name in keys
