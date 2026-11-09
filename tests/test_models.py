"""Unit tests for models and leakage-safe cross-validation.

Skipped automatically if scikit-learn is not installed.

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sepsis_icu.features import build_features
from sepsis_icu.metrics import auroc
from sepsis_icu.models import (
    build_target,
    cross_validated_predictions,
    feature_columns,
    make_gradient_boosting,
    make_logistic_regression,
)
from sepsis_icu.splits import make_patient_split, rows_in

NAN = np.nan


def make_data(n_patients=300, n_septic=60, seed=0):
    """Fake patient-hours: septic patients have higher HR and Temp once label is 1."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_patients):
        n = int(rng.integers(6, 12))
        y = np.zeros(n, dtype=int)
        if i < n_septic:
            y[int(rng.integers(1, n - 1)):] = 1
        hr = rng.normal(80 + 25 * y, 12)
        hr[rng.random(n) < 0.2] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": f"A_p{i:04d}",
                    "ICULOS": np.arange(1, n + 1),
                    "SepsisLabel": y,
                    "HR_ffill": hr,
                    "Temp_ffill": rng.normal(37 + 0.8 * y, 0.5),
                    "noise": rng.normal(size=n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def one_patient(label_start, n=10, pid="p1"):
    y = np.zeros(n, dtype=int)
    if label_start is not None:
        y[label_start:] = 1
    return pd.DataFrame({"patient_id": pid, "ICULOS": np.arange(1, n + 1), "SepsisLabel": y})


# ---------- build_target ----------

def test_default_horizon_reproduces_sepsis_label_exactly():
    df = make_data()
    assert (build_target(df).to_numpy() == df["SepsisLabel"].to_numpy()).all()


def test_longer_horizon_turns_positive_earlier():
    df = one_patient(label_start=6)  # label turns on at hour 7 -> onset hour 13
    h6 = build_target(df, horizon_hours=6)
    h9 = build_target(df, horizon_hours=9)
    assert h6[h6 == 1].index.min() == 6      # hour 7 (row 6)
    assert h9[h9 == 1].index.min() == 3      # hour 4 (row 3): 3 hours earlier
    assert h9.sum() == h6.sum() + 3


def test_zero_horizon_with_zero_shift_matches_label():
    df = make_data()
    t = build_target(df, horizon_hours=0, shift=0)
    assert (t.to_numpy() == df["SepsisLabel"].to_numpy()).all()


def test_zero_horizon_only_counts_from_onset():
    df = one_patient(label_start=6)  # onset at hour 13, after the 10-hour record
    assert build_target(df, horizon_hours=0).sum() == 0


def test_never_septic_patients_are_always_zero():
    df = one_patient(label_start=None)
    assert build_target(df, horizon_hours=24).sum() == 0


def test_target_aligns_with_original_row_order():
    df = make_data(n_patients=40, n_septic=10)
    shuffled = df.sample(frac=1, random_state=5)
    expected = build_target(df, horizon_hours=8).reindex(shuffled.index)
    assert (build_target(shuffled, horizon_hours=8).to_numpy() == expected.to_numpy()).all()


def test_build_target_rejects_bad_input():
    with pytest.raises(ValueError):
        build_target(one_patient(3), horizon_hours=-1)
    with pytest.raises(ValueError, match="SepsisLabel"):
        build_target(one_patient(3).drop(columns=["SepsisLabel"]))


# ---------- feature_columns ----------

def real_features():
    frames = []
    for pid in ("p1", "p2"):
        n = 6
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": pid,
                    "ICULOS": np.arange(1, n + 1),
                    "HR": [80, NAN, 90, NAN, NAN, 100],
                    "Temp": [NAN, 37.0, NAN, 38.0, NAN, NAN],
                }
            )
        )
    return build_features(pd.concat(frames, ignore_index=True), ["HR", "Temp"], windows=[3])


def test_latest_columns_are_only_the_forward_filled_values():
    assert feature_columns(real_features(), kind="latest") == ["HR_ffill", "Temp_ffill"]


def test_all_columns_with_and_without_missingness_indicators():
    feats = real_features()
    with_ind = feature_columns(feats, kind="all")
    without = feature_columns(feats, kind="all", include_missingness=False)
    assert len(with_ind) == 14       # 2 variables x (ffill + mean/min/max/delta + 2 indicators)
    assert len(without) == 10
    assert set(with_ind) - set(without) == {
        "HR_observed", "HR_hours_since", "Temp_observed", "Temp_hours_since"
    }
    assert "patient_id" not in with_ind and "ICULOS" not in with_ind


def test_time_column_is_excluded_unless_requested():
    feats = real_features()
    assert "ICULOS" not in feature_columns(feats)
    assert feature_columns(feats, include_time=True)[0] == "ICULOS"


def test_feature_columns_rejects_bad_requests():
    feats = real_features()
    with pytest.raises(ValueError):
        feature_columns(feats, kind="everything")
    with pytest.raises(ValueError, match="no feature"):
        feature_columns(feats[["patient_id", "ICULOS"]])


# ---------- models ----------

def split_xy(seed=0):
    df = make_data(seed=seed)
    X = df[["HR_ffill", "Temp_ffill", "noise"]]
    y = df["SepsisLabel"].to_numpy()
    # Alternate patients so both halves contain septic patients.
    train = rows_in(df["patient_id"], df["patient_id"].unique()[::2])
    return X, y, train


def test_logistic_regression_handles_missing_values_and_returns_probabilities():
    X, y, train = split_xy()
    model = make_logistic_regression().fit(X[train], y[train])
    p = model.predict_proba(X[~train])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert auroc(y[~train], p) > 0.85


def test_logistic_regression_imputer_is_fit_on_training_data_only():
    X, y, train = split_xy()
    model = make_logistic_regression().fit(X[train], y[train])
    train_medians = np.nanmedian(X[train].to_numpy(), axis=0)
    assert np.allclose(model.named_steps["impute"].statistics_, train_medians)
    # Predicting on other data does not change what was learned.
    model.predict_proba(X[~train])
    assert np.allclose(model.named_steps["impute"].statistics_, train_medians)


def test_logistic_regression_survives_an_all_missing_column():
    X, y, train = split_xy()
    X = X.copy()
    X["empty"] = np.nan
    model = make_logistic_regression().fit(X[train], y[train])
    assert np.isfinite(model.predict_proba(X[~train])).all()


def test_gradient_boosting_has_early_stopping_off_and_runs_every_iteration():
    X, y, train = split_xy()
    model = make_gradient_boosting(max_iter=15)
    assert model.early_stopping is False
    model.fit(X[train], y[train])
    assert model.n_iter_ == 15


def test_gradient_boosting_learns_the_signal_and_handles_missing_values():
    X, y, train = split_xy()
    model = make_gradient_boosting(max_iter=50).fit(X[train], y[train])
    p = model.predict_proba(X[~train])[:, 1]
    assert auroc(y[~train], p) > 0.85


def test_gradient_boosting_is_reproducible_with_a_fixed_seed():
    X, y, train = split_xy()
    a = make_gradient_boosting(max_iter=30, random_state=1).fit(X[train], y[train])
    b = make_gradient_boosting(max_iter=30, random_state=1).fit(X[train], y[train])
    assert np.array_equal(a.predict_proba(X[~train]), b.predict_proba(X[~train]))


# ---------- cross_validated_predictions ----------

def cv_setup():
    df = make_data()
    X = df[["HR_ffill", "Temp_ffill", "noise"]]
    y = df["SepsisLabel"].to_numpy()
    pids = df["patient_id"].to_numpy()
    split = make_patient_split(pids, y, holdout_fraction=0.2, n_splits=5)
    return df, X, y, pids, split


def test_oof_predictions_cover_development_rows_and_skip_the_holdout():
    _, X, y, pids, split = cv_setup()
    pred = cross_validated_predictions(X, y, pids, split, lambda: make_gradient_boosting(max_iter=20))
    holdout_rows = rows_in(pids, split.holdout_ids())
    assert np.isnan(pred[holdout_rows]).all()
    assert not np.isnan(pred[~holdout_rows]).any()
    assert auroc(y[~holdout_rows], pred[~holdout_rows]) > 0.85


class Spy:
    """Stand-in model that records which patients it was trained on."""

    fits = []

    def fit(self, X, y):
        Spy.fits.append(set(np.unique(X[:, 0]).astype(int).tolist()))
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate)
        return np.column_stack([1 - p, p])


def test_no_model_ever_trains_on_its_validation_patients_or_the_holdout():
    df, _, y, pids, split = cv_setup()
    codes = pd.factorize(df["patient_id"])[0]
    code_of = dict(zip(pids, codes))
    X = np.column_stack([codes, df["noise"].to_numpy()])

    Spy.fits = []
    cross_validated_predictions(X, y, pids, split, Spy)

    assert len(Spy.fits) == 5
    holdout_codes = {code_of[p] for p in split.holdout_ids()}
    for k, trained_on in enumerate(Spy.fits):
        _, val_ids = split.train_val_ids(k)
        val_codes = {code_of[p] for p in val_ids}
        assert not trained_on & val_codes
        assert not trained_on & holdout_codes
        assert trained_on  # it did train on something


def test_a_fresh_model_is_made_for_every_fold():
    _, X, y, pids, split = cv_setup()
    made = []

    def factory():
        made.append(1)
        return make_logistic_regression()

    cross_validated_predictions(X, y, pids, split, factory)
    assert len(made) == 5


def test_oof_predictions_are_reproducible():
    _, X, y, pids, split = cv_setup()
    fn = lambda: make_gradient_boosting(max_iter=20, random_state=3)
    a = cross_validated_predictions(X, y, pids, split, fn)
    b = cross_validated_predictions(X, y, pids, split, fn)
    assert np.array_equal(a, b, equal_nan=True)


def test_cv_rejects_misaligned_inputs():
    _, X, y, pids, split = cv_setup()
    with pytest.raises(ValueError, match="same number of rows"):
        cross_validated_predictions(X, y[:-1], pids, split, make_logistic_regression)


def test_cv_rejects_rows_from_patients_outside_the_split():
    df, X, y, pids, split = cv_setup()
    other = pids.copy()
    other[0] = "B_p9999"  # a patient the split has never seen (for example hospital B)
    with pytest.raises(ValueError, match="not in the split"):
        cross_validated_predictions(X, y, other, split, make_logistic_regression)


def test_cv_rejects_a_training_set_with_one_class():
    _, X, y, pids, split = cv_setup()
    with pytest.raises(ValueError, match="one class"):
        cross_validated_predictions(X, np.zeros(len(y), dtype=int), pids, split, make_logistic_regression)
