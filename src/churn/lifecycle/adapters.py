"""Small model adapters for the lifecycle / retrain path."""

from __future__ import annotations


class ColumnSubsetModel:
    """Wrap a model fit on a column subset so ``predict_proba`` takes the FULL design matrix.

    The retrain path fits champions and challengers on different feature subsets but scores them all
    on the same full holdout matrix; this adapts a subset-fit model to that matrix by slicing to the
    columns it was trained on. Shared by the orchestration retrain branch and its test, so the two
    never drift out of sync.
    """

    def __init__(self, model, cols):
        self.model = model
        self.cols = cols

    def predict_proba(self, x):
        return self.model.predict_proba(x[:, self.cols])
