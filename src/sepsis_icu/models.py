"""Models and leakage-safe cross-validation (study_design.md, sections 1, 4, 5 and 8).

Requires scikit-learn >= 1.2 (add it to your dependencies).

Contents:
  build_target                     target "sepsis within the next H hours" from SepsisLabel
  feature_columns                  pick model inputs from the output of build_features
  make_logistic_regression         baseline: logistic regression on latest values
  make_gradient_boosting           main model: gradient boosting on rolling-window features
  cross_validated_predictions      out-of-fold predictions using patient-level folds

Leakage rules this module follows:
  * Imputers and scalers live inside the model pipeline, so in cross-validation
    they are fit on training patients only.
  * Gradient boosting has early stopping switched OFF. scikit-learn's
    HistGradientBoostingClassifier turns early stopping on by itself for large
    datasets and holds out random ROWS for validation, which would put the same
    patient on both sides inside a single fit. Choose the number of iterations
    with the patient-level folds instead.
  * No class weights are used, so predicted probabilities stay interpretable
    (and calibration can be assessed honestly). Recalibrate afterwards if needed.
  * Folds come from sepsis_icu.splits (patient-level). The holdout patients and
    hospital B are never passed to a fit.

The time column (ICULOS) is not a model input unless you ask for it
(include_time=True in feature_columns), so the ICU-length-of-stay-only
baseline stays a clean comparison.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sepsis_icu.onset import LABEL_SHIFT_HOURS
from sepsis_icu.splits import PatientSplit, assert_disjoint, rows_in

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_TIME_COLUMN = "ICULOS"
DEFAULT_LABEL_COLUMN = "SepsisLabel"

MISSINGNESS_SUFFIXES = ("_observed", "_hours_since")


# ---------- target ----------

def build_target(
    df: pd.DataFrame,
    horizon_hours: float = 6.0,
    shift: float = LABEL_SHIFT_HOURS,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.Series:
    """Target for each patient-hour: sepsis within the next `horizon_hours`.

    y = 1 when the reconstructed onset (first SepsisLabel == 1 hour + shift) is at
    most `horizon_hours` after this hour, or has already happened. Hours at or
    after onset stay positive (the patient is already septic), matching the
    Challenge labels. Patients who never become septic are always 0.

    With the default horizon of 6 and shift of 6 this reproduces SepsisLabel
    exactly, because the Challenge label already turns on 6 hours before onset.
    Log the horizon you use in the decisions log.

    Returns a 0/1 Series with the same index as df.
    """
    if horizon_hours < 0:
        raise ValueError("horizon_hours must be non-negative")
    missing = [c for c in (id_col, time_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    first_label_hour = df.loc[df[label_col] == 1].groupby(id_col)[time_col].min()
    onset = df[id_col].map(first_label_hour) + shift  # NaN for never-septic patients
    target = (df[time_col] >= onset - horizon_hours).astype(int)  # NaN compares False -> 0
    return target.rename("target")


# ---------- features ----------

def feature_columns(
    features: pd.DataFrame,
    kind: str = "all",
    include_missingness: bool = True,
    include_time: bool = False,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> List[str]:
    """Model input columns from the output of sepsis_icu.features.build_features.

    kind="latest": only the last-known-value columns ("*_ffill"), for the
        logistic regression baseline.
    kind="all": every feature column (latest values, rolling windows, changes,
        and missingness indicators), for the main model.
    include_missingness=False drops "*_observed" and "*_hours_since" so results
        can be reported with and without missingness indicators.
    include_time=True adds the time column (ICULOS) as a feature.
    """
    if kind not in ("latest", "all"):
        raise ValueError("kind must be 'latest' or 'all'")

    cols = [c for c in features.columns if c not in (id_col, time_col)]
    if kind == "latest":
        cols = [c for c in cols if c.endswith("_ffill")]
    if not include_missingness:
        cols = [c for c in cols if not c.endswith(MISSINGNESS_SUFFIXES)]
    if include_time:
        cols = [time_col] + cols
    if not cols:
        raise ValueError("no feature columns selected")
    return cols


# ---------- models ----------

def make_logistic_regression(
    C: float = 1.0, max_iter: int = 1000, random_state: int = 42
) -> Pipeline:
    """Baseline: median-impute, standardize, logistic regression.

    The imputer and scaler are part of the pipeline, so they are fit on whatever
    training data the pipeline is fit on, and never on validation or test data.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)),
        ]
    )


def make_gradient_boosting(
    max_iter: int = 200,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """Main model: histogram gradient boosting, which handles missing values itself.

    Early stopping is explicitly OFF (see the module notes). Tune max_iter,
    learning_rate and the tree settings with the patient-level folds.
    """
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=random_state,
    )


# ---------- cross-validation ----------

def cross_validated_predictions(
    X,
    y: Sequence,
    patient_ids: Sequence,
    split: PatientSplit,
    make_model: Callable[[], object],
) -> np.ndarray:
    """Out-of-fold predicted probabilities using the patient-level folds in `split`.

    For each fold, a NEW model from make_model() is fit on the training patients'
    rows and predicts the validation patients' rows. Every development row gets
    exactly one prediction, made by a model that never saw that patient.
    Holdout rows are never used and get NaN.

    X: row-level features (DataFrame or array), aligned with y and patient_ids.
    make_model: zero-argument function returning a fresh, unfitted estimator
        with fit and predict_proba.
    """
    y = np.asarray(y)
    pids = np.asarray(patient_ids)
    if not (len(X) == len(y) == len(pids)):
        raise ValueError("X, y and patient_ids must have the same number of rows")
    if not rows_in(pids, split.patient_ids).all():
        raise ValueError("some rows belong to patients that are not in the split")

    predictions = np.full(len(y), np.nan)
    holdout_ids = split.holdout_ids()
    for k in range(split.n_splits):
        train_ids, val_ids = split.train_val_ids(k)
        assert_disjoint(train=train_ids, validation=val_ids, holdout=holdout_ids)

        train = rows_in(pids, train_ids)
        val = rows_in(pids, val_ids)
        if y[train].min() == y[train].max():
            raise ValueError(f"fold {k}: training rows contain only one class")

        model = make_model()
        model.fit(X[train], y[train])
        predictions[val] = model.predict_proba(X[val])[:, 1]
    return predictions
