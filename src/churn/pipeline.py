"""Assemble the leak-free modelling pipeline.

Structure::

    DomainRepair → FeatureEngineer → ColumnTransformer(
        numeric:     Winsorizer(spend only) → SimpleImputer(median, +indicator) → StandardScaler
        categorical: SimpleImputer(most_frequent) → OneHotEncoder(ignore unknown)
    ) → estimator

Everything that learns from data sits inside this object, so wrapping it in
cross-validation guarantees each fold's statistics come from its training rows
only. The exact same fitted object is reused at scoring time → no train/serve skew.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .cleaning import DomainRepair, Winsorizer
from .features import FeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("winsorize", Winsorizer(columns=config.WINSORIZE_COLUMNS)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Wrap an estimator with the full leak-free preprocessing chain."""
    return Pipeline(
        steps=[
            ("repair", DomainRepair()),
            ("features", FeatureEngineer()),
            ("pre", build_preprocessor()),
            ("clf", model),
        ]
    )


def candidate_models(seed: int = config.SEED) -> dict:
    """The model bake-off. Linear baseline + two nonlinear challengers.

    The target is balanced (~48% churn), so we deliberately do NOT resample or
    set class_weight — the 'imbalance' in this dataset is in subscription_plan,
    not the label.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
