"""Customer churn prediction — leak-free, reproducible modelling package.

The public surface is intentionally small; the entry point ``churn_prediction.py``
orchestrates these modules into an end-to-end run.
"""

__all__ = [
    "config",
    "data",
    "cleaning",
    "features",
    "pipeline",
    "evaluate",
    "interpret",
    "registry",
    "scoring",
    "train",
    "plots",
]
