"""Type-aware drift detection (ENHANCEMENT_PLAN.md Sec 4).

Per-feature detectors chosen by TYPE, not one-size-fits-all:

- **numeric** -- PSI **and** a two-sample KS test; a feature alarms only when **both** fire (PSI
  over threshold AND KS significant), trading a little recall for far fewer false alarms.
- **categorical** -- PSI **and** a chi-square test of the contingency table (Cramér's V as the
  effect size). This is the fix for ``monitoring.py`` reporting a meaningless ``KS = NaN`` on
  categoricals -- KS is undefined on unordered categories; chi-square is the right instrument.

Plus a **domain-classifier** alarm: if a classifier can tell reference from current rows
(cross-validated ROC-AUC well above 0.5), the joint distribution has shifted even when no single
marginal trips -- the multivariate covariate-shift signal single-feature tests miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config, monitoring

PSI_THRESHOLD = 0.2  # PSI > 0.2 is the classic "investigate" band
KS_ALPHA = 0.05
CHI2_ALPHA = 0.05
DOMAIN_AUC_ALARM = 0.65  # domain classifier this discriminative => covariate shift


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    kind: str  # "numeric" | "categorical"
    psi: float
    stat: float  # KS statistic (numeric) | Cramer's V (categorical)
    p_value: float
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    features: list[FeatureDrift]
    domain_classifier_auc: float
    covariate_shift: bool

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]


def _numeric_drift(name, ref, cur, psi_threshold: float, alpha: float) -> FeatureDrift:
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(dtype="float64")
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(dtype="float64")
    psi = monitoring.numeric_psi(pd.Series(r), pd.Series(c))
    if len(r) == 0 or len(c) == 0:
        return FeatureDrift(name, "numeric", psi, 0.0, 1.0, False)
    ks = ks_2samp(r, c)
    drifted = bool(psi > psi_threshold and ks.pvalue < alpha)  # BOTH fire
    return FeatureDrift(name, "numeric", psi, float(ks.statistic), float(ks.pvalue), drifted)


def _categorical_drift(name, ref, cur, psi_threshold: float, alpha: float) -> FeatureDrift:
    psi = monitoring.categorical_psi(ref, cur)
    ref_c = ref.astype("object").where(ref.notna(), monitoring.MISSING_TOKEN)
    cur_c = cur.astype("object").where(cur.notna(), monitoring.MISSING_TOKEN)
    levels = sorted(set(ref_c.unique()) | set(cur_c.unique()), key=str)
    ref_counts = ref_c.value_counts().reindex(levels, fill_value=0).to_numpy()
    cur_counts = cur_c.value_counts().reindex(levels, fill_value=0).to_numpy()
    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]  # drop empty levels (chi2 needs positive margins)
    if table.shape[1] < 2:
        return FeatureDrift(name, "categorical", psi, 0.0, 1.0, False)
    chi2, p_value, _, _ = chi2_contingency(table)
    cramers_v = float(np.sqrt(chi2 / table.sum()))  # rows=2 -> min(r-1,c-1) collapses to 1
    drifted = bool(psi > psi_threshold and p_value < alpha)  # BOTH fire
    return FeatureDrift(name, "categorical", psi, cramers_v, float(p_value), drifted)


def domain_classifier_auc(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int = config.SEED,
) -> float:
    """Cross-validated ROC-AUC of a classifier separating reference (0) from current (1) rows."""
    ref = reference[numeric + categorical].copy()
    cur = current[numeric + categorical].copy()
    x = pd.concat([ref, cur], ignore_index=True)
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))]).astype(int)
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    *,
    psi_threshold: float = PSI_THRESHOLD,
    ks_alpha: float = KS_ALPHA,
    chi2_alpha: float = CHI2_ALPHA,
    domain_auc_alarm: float = DOMAIN_AUC_ALARM,
    seed: int = config.SEED,
) -> DriftReport:
    """Type-aware per-feature drift + a domain-classifier covariate-shift alarm."""
    if numeric_features is None and categorical_features is None:
        is_num = pd.api.types.is_numeric_dtype
        numeric_features = [c for c in reference.columns if is_num(reference[c])]
        categorical_features = [c for c in reference.columns if c not in numeric_features]
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    features = [
        _numeric_drift(c, reference[c], current[c], psi_threshold, ks_alpha)
        for c in numeric_features
    ]
    features += [
        _categorical_drift(c, reference[c], current[c], psi_threshold, chi2_alpha)
        for c in categorical_features
    ]
    dauc = domain_classifier_auc(
        reference, current, numeric_features, categorical_features, seed=seed
    )
    return DriftReport(features, dauc, covariate_shift=dauc > domain_auc_alarm)
