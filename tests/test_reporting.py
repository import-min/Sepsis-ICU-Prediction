"""Unit tests for the reporting tables and the saved model and calibrator files.

The model-saving tests are skipped automatically if scikit-learn is not installed.

Run from the repo root:  pytest -q
"""

import json
import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from sepsis_icu.recalibration import InterceptShift, IsotonicCalibrator, PlattScaler, expit, logit
from sepsis_icu.reporting import (
    analysis_counts,
    calibrator_from_dict,
    calibrator_to_dict,
    characteristics_shift,
    characteristics_table,
    feature_inventory,
    load_calibrator,
    load_model,
    save_calibrator,
    save_model,
)
from sepsis_icu.splits import make_patient_split

NAN = np.nan


# ---------- analysis_counts ----------

def counts_data(n_patients=100, n_septic=20, prefix="A", seed=0):
    rng = np.random.default_rng(seed)
    frames, ys = [], []
    for i in range(n_patients):
        n = int(rng.integers(6, 12))
        label = np.zeros(n, dtype=int)
        if i < n_septic:
            label[int(rng.integers(1, n - 1)):] = 1
        frames.append(pd.DataFrame({"patient_id": f"{prefix}_p{i:03d}", "ICULOS": np.arange(1, n + 1), "SepsisLabel": label}))
        ys.append(label)  # the target equals the label here
    return pd.concat(frames, ignore_index=True), np.concatenate(ys)


def counts_row(table, name):
    return table.set_index("analysis").loc[name]


def test_analysis_counts_are_consistent_across_the_split():
    df, y = counts_data()
    split = make_patient_split(df["patient_id"], df["SepsisLabel"], holdout_fraction=0.2, n_splits=4)
    t = analysis_counts(df, y, split)
    assert list(t.columns) == ["analysis", "patients", "septic_patients", "hours", "positive_hours"]
    all_, dev, hold = (counts_row(t, n) for n in ("hospital_a_all", "hospital_a_development", "hospital_a_holdout"))
    assert all_["patients"] == 100 and all_["septic_patients"] == 20
    assert all_["hours"] == len(df) and all_["positive_hours"] == y.sum()
    for column in ("patients", "septic_patients", "hours", "positive_hours"):
        assert dev[column] + hold[column] == all_[column]

    for k in range(4):
        train, val = counts_row(t, f"fold_{k}_training"), counts_row(t, f"fold_{k}_validation")
        assert train["patients"] + val["patients"] == dev["patients"]
        assert train["hours"] + val["hours"] == dev["hours"]
    validation_patients = sum(counts_row(t, f"fold_{k}_validation")["patients"] for k in range(4))
    assert validation_patients == dev["patients"]   # every development patient is validated once


def test_analysis_counts_includes_hospital_b_when_given():
    df, y = counts_data()
    split = make_patient_split(df["patient_id"], df["SepsisLabel"], holdout_fraction=0.2, n_splits=3)
    df_b, y_b = counts_data(n_patients=40, n_septic=10, prefix="B", seed=5)
    t = analysis_counts(df, y, split, external=(df_b, y_b))
    b = counts_row(t, "hospital_b")
    assert (b["patients"], b["septic_patients"], b["hours"]) == (40, 10, len(df_b))
    assert "hospital_b" not in analysis_counts(df, y, split)["analysis"].tolist()


def test_analysis_counts_rejects_a_target_of_the_wrong_length():
    df, y = counts_data()
    split = make_patient_split(df["patient_id"], df["SepsisLabel"], holdout_fraction=0.2, n_splits=3)
    with pytest.raises(ValueError):
        analysis_counts(df, y[:-1], split)


def test_positive_hours_follow_the_target_not_the_label():
    df, y = counts_data()
    split = make_patient_split(df["patient_id"], df["SepsisLabel"], holdout_fraction=0.2, n_splits=3)
    shifted = np.zeros_like(y)   # a target with no positive hours
    t = analysis_counts(df, shifted, split)
    assert (t["positive_hours"] == 0).all() and counts_row(t, "hospital_a_all")["septic_patients"] == 20


# ---------- characteristics ----------

def patients(ages, sexes, units, stays, missing, septic):
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(len(ages))],
            "age": ages, "sex": sexes, "unit": units, "stay_hours": stays,
            "missing_fraction": missing, "ever_septic": septic,
        }
    )


def small_patients():
    return patients(
        ages=[40.0, 50.0, 60.0, NAN],
        sexes=["female", "male", "male", "unknown"],
        units=["MICU", "MICU", "SICU", "unknown"],
        stays=[10, 20, 30, 40],
        missing=[0.2, 0.4, 0.6, 0.8],
        septic=[True, False, False, True],
    )


def test_characteristics_table_values():
    t = characteristics_table({"d": small_patients()})
    assert list(t.columns) == ["dataset", "variable", "level", "n", "value", "q25", "q75"]

    def row(variable, level=""):
        return t[(t["variable"] == variable) & (t["level"] == level)].iloc[0]

    assert row("patients")["n"] == 4
    age = row("age")
    assert (age["n"], age["value"], age["q25"], age["q75"]) == (3, 50.0, 45.0, 55.0)   # the missing age is not counted
    assert row("stay_hours")["value"] == 25.0
    assert row("sex", "male")["n"] == 2 and row("sex", "male")["value"] == 50.0
    assert row("sex", "female")["value"] == 25.0
    assert row("unit", "SICU")["n"] == 1
    assert row("ever_septic", "yes")["n"] == 2 and row("ever_septic", "yes")["value"] == 50.0


def test_characteristics_table_keeps_the_same_rows_for_every_dataset():
    only_male = patients([30.0], ["male"], ["MICU"], [10], [0.1], [False])
    t = characteristics_table({"a": small_patients(), "b": only_male})
    rows_a = t[t["dataset"] == "a"][["variable", "level"]].reset_index(drop=True)
    rows_b = t[t["dataset"] == "b"][["variable", "level"]].reset_index(drop=True)
    assert rows_a.equals(rows_b)
    zero = t[(t["dataset"] == "b") & (t["variable"] == "sex") & (t["level"] == "female")].iloc[0]
    assert zero["n"] == 0 and zero["value"] == 0.0


def test_characteristics_table_adds_levels_it_has_not_heard_of():
    odd = patients([30.0], ["nonbinary"], ["MICU"], [10], [0.1], [False])
    t = characteristics_table({"a": small_patients(), "b": odd})
    assert "nonbinary" in set(t.loc[t["variable"] == "sex", "level"])


def test_characteristics_table_handles_an_empty_dataset():
    empty = small_patients().iloc[0:0]
    t = characteristics_table({"e": empty})
    assert t.loc[t["variable"] == "patients", "n"].iloc[0] == 0
    assert np.isnan(t.loc[(t["variable"] == "sex") & (t["level"] == "male"), "value"].iloc[0])


def test_characteristics_shift_numeric_and_categorical():
    a = patients([40.0, 50.0, 60.0], ["female", "male", "male", ], ["MICU"] * 3, [10, 20, 30], [0.1, 0.2, 0.3], [True, False, False])
    b = patients([60.0, 70.0, 80.0], ["female", "female", "female"], ["SICU"] * 3, [10, 20, 30], [0.1, 0.2, 0.3], [False, False, False])
    t = characteristics_shift(a, b).set_index(["variable", "level"])
    assert list(characteristics_shift(a, b).columns) == [
        "variable", "level", "hospital_a", "hospital_b", "standardized_difference"
    ]
    age = t.loc[("age", "")]
    assert (age["hospital_a"], age["hospital_b"]) == (50.0, 70.0)
    assert age["standardized_difference"] == pytest.approx(-2.0)        # (50 - 70) / sd 10
    assert t.loc[("stay_hours", ""), "standardized_difference"] == pytest.approx(0.0)
    # Female: 1 of 3 in A, 3 of 3 in B.
    female = t.loc[("sex", "female")]
    assert female["hospital_a"] == pytest.approx(1 / 3) and female["hospital_b"] == 1.0
    assert female["standardized_difference"] == pytest.approx(-2.0)   # (1/3 - 1) / 0.3333
    septic = t.loc[("ever_septic", "yes")]
    assert septic["hospital_a"] == pytest.approx(1 / 3) and septic["hospital_b"] == 0.0
    assert septic["standardized_difference"] == pytest.approx(1.0)     # (1/3 - 0) / 0.3333


def test_characteristics_shift_is_undefined_when_there_is_no_spread():
    same = patients([50.0, 50.0], ["male", "male"], ["MICU", "MICU"], [10, 10], [0.1, 0.1], [False, False])
    t = characteristics_shift(same, same).set_index(["variable", "level"])
    assert np.isnan(t.loc[("age", ""), "standardized_difference"])
    assert np.isnan(t.loc[("sex", "male"), "standardized_difference"])


def test_characteristics_shift_custom_labels():
    t = characteristics_shift(small_patients(), small_patients(), label_a="dev", label_b="ext")
    assert {"dev", "ext"} <= set(t.columns)


# ---------- feature_inventory ----------

def test_feature_inventory_parses_every_kind_of_feature():
    columns = [
        "HR_ffill", "HR_mean3h", "HR_min12h", "HR_max6h", "HR_delta3h",
        "HR_observed", "HR_hours_since", "Bilirubin_direct_ffill", "Bilirubin_direct_mean6h",
        "Bilirubin_direct_hours_since", "ICULOS",
    ]
    t = feature_inventory({"m": columns}).set_index("feature")
    assert t.loc["HR_ffill", "variable"] == "HR" and t.loc["HR_ffill", "kind"] == "latest value"
    assert np.isnan(t.loc["HR_ffill", "window_hours"])
    assert t.loc["HR_mean3h", "kind"] == "mean over window" and t.loc["HR_mean3h", "window_hours"] == 3
    assert t.loc["HR_min12h", "kind"] == "minimum over window" and t.loc["HR_min12h", "window_hours"] == 12
    assert t.loc["HR_max6h", "kind"] == "maximum over window"
    assert t.loc["HR_delta3h", "kind"] == "change over window"
    assert t.loc["HR_observed", "is_missingness_indicator"] and t.loc["HR_hours_since", "is_missingness_indicator"]
    assert not t.loc["HR_mean3h", "is_missingness_indicator"]
    # Variable names that contain underscores are kept whole.
    assert t.loc["Bilirubin_direct_mean6h", "variable"] == "Bilirubin_direct"
    assert t.loc["Bilirubin_direct_hours_since", "variable"] == "Bilirubin_direct"
    assert t.loc["ICULOS", "kind"] == "other" and t.loc["ICULOS", "variable"] == "ICULOS"


def test_feature_inventory_lists_every_model():
    t = feature_inventory({"a": ["HR_ffill"], "b": ["HR_ffill", "HR_mean3h"]})
    assert t.groupby("model").size().to_dict() == {"a": 1, "b": 2}
    assert list(t.columns) == ["model", "feature", "variable", "kind", "window_hours", "is_missingness_indicator"]


def test_feature_inventory_matches_what_build_features_makes():
    from sepsis_icu.features import build_features

    df = pd.DataFrame({"patient_id": "p", "ICULOS": range(1, 7), "HR": [80, NAN, 90, 95, NAN, 100]})
    columns = [c for c in build_features(df, ["HR"], windows=[3]).columns if c not in ("patient_id", "ICULOS")]
    t = feature_inventory({"m": columns})
    assert (t["kind"] != "other").all() and set(t["variable"]) == {"HR"}


# ---------- saved models ----------

def fitted_models():
    pytest.importorskip("sklearn")
    from sepsis_icu.models import make_gradient_boosting, make_logistic_regression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) + rng.normal(0, 0.5, 400) > 0.3).astype(int)
    return X, {
        "logistic": make_logistic_regression().fit(X, y),
        "boosting": make_gradient_boosting(max_iter=20).fit(X, y),
    }


@pytest.mark.parametrize("which", ["logistic", "boosting"])
def test_a_saved_model_predicts_exactly_the_same_after_loading(tmp_path, which):
    X, models = fitted_models()
    model = models[which]
    meta = {"model": which, "feature_columns": ["a", "b", "c", "d", "e"], "threshold": 0.37}
    model_path, meta_path = save_model(model, str(tmp_path / "sub" / which), meta)
    assert model_path.name == f"{which}.joblib" and meta_path.name == f"{which}.json"

    loaded, metadata = load_model(str(tmp_path / "sub" / which))
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert metadata["feature_columns"] == ["a", "b", "c", "d", "e"] and metadata["threshold"] == 0.37
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(metadata["versions"])
    assert "saved_utc" in metadata


def test_models_can_be_loaded_with_or_without_the_extension(tmp_path):
    X, models = fitted_models()
    save_model(models["logistic"], str(tmp_path / "m.joblib"))
    loaded, _ = load_model(str(tmp_path / "m"))
    assert np.array_equal(loaded.predict_proba(X), models["logistic"].predict_proba(X))


def test_loading_warns_when_the_scikit_learn_version_changed(tmp_path):
    _, models = fitted_models()
    _, meta_path = save_model(models["logistic"], str(tmp_path / "m"))
    meta = json.loads(meta_path.read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(str(tmp_path / "m"))


def test_loading_a_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "nothing"))


def test_a_model_without_a_metadata_file_still_loads(tmp_path):
    X, models = fitted_models()
    _, meta_path = save_model(models["logistic"], str(tmp_path / "m"))
    meta_path.unlink()
    loaded, metadata = load_model(str(tmp_path / "m"))
    assert metadata == {} and loaded.predict_proba(X).shape == (400, 2)


# ---------- saved calibrators ----------

def fitted_calibrators():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 5000)
    y = (rng.random(5000) < expit(0.6 * logit(p) + 0.4)).astype(int)
    return p, {
        "platt": PlattScaler().fit(p, y),
        "intercept_shift": InterceptShift().fit(p, y),
        "isotonic": IsotonicCalibrator().fit(p, y),
    }


@pytest.mark.parametrize("kind", ["platt", "intercept_shift", "isotonic"])
def test_calibrators_survive_a_round_trip_through_json(tmp_path, kind):
    p, calibrators = fitted_calibrators()
    original = calibrators[kind]
    path = save_calibrator(original, str(tmp_path / "cal" / f"{kind}.json"), metadata={"fit_on": "development"})
    saved = json.loads(path.read_text())
    assert saved["type"] == kind and saved["metadata"] == {"fit_on": "development"}

    loaded = load_calibrator(str(path))
    assert type(loaded) is type(original)
    grid = np.linspace(0, 1, 201)
    assert np.array_equal(loaded.transform(grid), original.transform(grid))
    assert np.array_equal(loaded.transform(p), original.transform(p))


@pytest.mark.parametrize("calibrator", [PlattScaler, InterceptShift, IsotonicCalibrator])
def test_an_unfitted_calibrator_cannot_be_saved(tmp_path, calibrator):
    with pytest.raises(ValueError, match="not been fit"):
        save_calibrator(calibrator(), str(tmp_path / "x.json"))


def test_unknown_calibrators_are_rejected():
    with pytest.raises(ValueError, match="unknown calibrator type"):
        calibrator_to_dict(object())
    with pytest.raises(ValueError, match="unknown calibrator type"):
        calibrator_from_dict({"type": "magic"})
    with pytest.raises(ValueError):
        calibrator_from_dict({})


def test_calibrator_to_dict_is_plain_json():
    _, calibrators = fitted_calibrators()
    for calibrator in calibrators.values():
        json.dumps(calibrator_to_dict(calibrator))   # must not raise
