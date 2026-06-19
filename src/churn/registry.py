"""Self-contained model registry: a run directory with the pipeline(s) + metadata.json."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
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


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_run(model, metadata: dict, base_model=None, base_dir=config.MODELS_DIR) -> Path:
    created = utc_now()
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
        "git_sha": git_sha(),
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "has_base_linear": base_model is not None,
        **metadata,
    }
    (run_dir / METADATA_FILE).write_text(json.dumps(to_jsonable(full_meta), indent=2))
    return run_dir


def latest_run_dir(base_dir=config.MODELS_DIR) -> Path:
    base = Path(base_dir)
    runs = [d for d in base.glob("*") if (d / PIPELINE_FILE).exists()]
    if not runs:
        raise FileNotFoundError(f"No saved runs found under {base}.")
    return max(runs, key=lambda d: d.stat().st_mtime)


@dataclass(frozen=True)
class LoadedModel:
    """A loaded run that owns its own cutpoints + base linear and can score a frame.

    Callers used to receive a raw (model, meta, base) tuple and each re-reach into
    meta["tier_cutpoints"] before scoring; that key path now lives here alone.
    """

    model: object
    metadata: dict
    base_linear: object | None

    @property
    def run_id(self) -> str | None:
        return self.metadata.get("run_id")

    @property
    def cutpoints(self) -> dict:
        return self.metadata["tier_cutpoints"]

    def score(self, df, k: int = 3):
        # Local import: registry is the lower layer; scoring imports registry.
        from .scoring import score_frame

        cut = self.cutpoints
        return score_frame(
            df, self.model, cut["t_star"], cut["t_mid"], base_linear=self.base_linear, k=k
        )


def load_run(run_dir=None) -> LoadedModel:
    run_dir = Path(run_dir) if run_dir is not None else latest_run_dir()
    model = joblib.load(run_dir / PIPELINE_FILE)
    metadata = json.loads((run_dir / METADATA_FILE).read_text())
    base = joblib.load(run_dir / BASE_FILE) if (run_dir / BASE_FILE).exists() else None
    return LoadedModel(model, metadata, base)
