"""Lightweight, self-contained model registry.

A run is a directory under ``models/`` holding the fitted pipeline(s) plus a
``metadata.json`` capturing everything needed to reproduce and trust the model:
params, metrics, the cost-based threshold + tier cutpoints, the data hash, and
library/git versions. (MLflow is the production swap-in; this proves the concept
without standing up a server for an 1,800-row CSV.)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config

PIPELINE_FILE = "pipeline.joblib"
BASE_FILE = "base_linear.joblib"
METADATA_FILE = "metadata.json"


def data_sha256(path=config.DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        # git absent (OSError) or non-zero exit / not a repo (SubprocessError).
        return None


def _now() -> datetime:
    return datetime.now(UTC)


def _jsonable(obj):
    """Recursively coerce numpy types so json.dump never chokes."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_run(model, metadata: dict, base_model=None, base_dir=config.MODELS_DIR) -> Path:
    created = _now()
    model_name = metadata.get("model_name", "model")
    run_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}_{model_name}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / PIPELINE_FILE)
    if base_model is not None:
        joblib.dump(base_model, run_dir / BASE_FILE)

    full_meta = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "data_sha256": data_sha256(),
        "git_sha": _git_sha(),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "has_base_linear": base_model is not None,
        **metadata,
    }
    (run_dir / METADATA_FILE).write_text(json.dumps(_jsonable(full_meta), indent=2))
    return run_dir


def latest_run_dir(base_dir=config.MODELS_DIR) -> Path:
    base = Path(base_dir)
    runs = [d for d in base.glob("*") if (d / PIPELINE_FILE).exists()]
    if not runs:
        raise FileNotFoundError(f"No saved runs found under {base}.")
    return max(runs, key=lambda d: d.stat().st_mtime)


def load_run(run_dir=None):
    """Return ``(model, metadata, base_model_or_None)`` for a run (latest by default)."""
    run_dir = Path(run_dir) if run_dir is not None else latest_run_dir()
    model = joblib.load(run_dir / PIPELINE_FILE)
    metadata = json.loads((run_dir / METADATA_FILE).read_text())
    base = joblib.load(run_dir / BASE_FILE) if (run_dir / BASE_FILE).exists() else None
    return model, metadata, base
