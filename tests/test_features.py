"""Unit tests for leakage-safe features.

The most important test is `test_features_at_t_ignore_the_future`: it proves
that changing or deleting data after hour T does not change any feature at or
before hour T.

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
from pandas.testing import assert_frame_equal

from sepsis_icu.features import (
    apply_medians,
    build_features,
    fit_medians,
    forward_fill,
    hours_since_measured,
)

NAN = np.nan


def one_patient(values, pid="p1", start=1, col="HR"):
    return pd.DataFrame(
        {
            "patient_id": pid,
            "ICULOS": range(start, start + len(values)),
            col: values,
        }
    )


def random_cohort(n_patients=8, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_patients):
        n = int(rng.integers(15, 30))
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": f"p{i}",
                    "ICULOS": range(1, n + 1),
                    "HR": np.where(rng.random(n) < 0.5, np.nan, rng.normal(85, 15, n)),
                    "Temp": np.where(rng.random(n) < 0.7, np.nan, rng.normal(37, 0.6, n)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------- forward_fill ----------

def test_forward_fill_is_forward_only_and_stays_within_patient():
    df = pd.concat(
        [
            one_patient([NAN, 80, NAN, 90], pid="p1"),
            one_patient([NAN, NAN, 70], pid="p2"),
        ],
        ignore_index=True,
    )
    out = forward_fill(df, ["HR"])["HR"].tolist()
    # p1: leading NaN stays NaN (no backward fill); p2 does not inherit p1's 90.
    expected = [NAN, 80, 80, 90, NAN, NAN, 70]
    assert np.allclose(out, expected, equal_nan=True)


def test_forward_fill_requires_sorted_rows():
    df = one_patient([1, 2, 3]).iloc[[2, 0, 1]]
    with pytest.raises(ValueError, match="sorted"):
        forward_fill(df, ["HR"])


# ---------- rolling features ----------

def test_rolling_and_delta_values_are_correct():
    df = one_patient([1, 2, 3, 4, 5])
    f = build_features(df, ["HR"], windows=[3], missingness_indicators=False)
    assert np.allclose(f["HR_ffill"], [1, 2, 3, 4, 5])
    assert np.allclose(f["HR_mean3h"], [1, 1.5, 2, 3, 4])
    assert np.allclose(f["HR_min3h"], [1, 1, 1, 2, 3])
    assert np.allclose(f["HR_max3h"], [1, 2, 3, 4, 5])
    assert np.allclose(f["HR_delta3h"], [NAN, NAN, NAN, 3, 3], equal_nan=True)


def test_windows_do_not_cross_patients():
    df = pd.concat(
        [one_patient([100, 100, 100], pid="p1"), one_patient([1, 1, 1], pid="p2")],
        ignore_index=True,
    )
    f = build_features(df, ["HR"], windows=[3], missingness_indicators=False)
    p2 = f[f["patient_id"] == "p2"]
    assert np.allclose(p2["HR_mean3h"], [1, 1, 1])
    assert np.allclose(p2["HR_delta3h"], [NAN, NAN, NAN], equal_nan=True)


# ---------- missingness indicators ----------

def test_missingness_indicators_values():
    df = one_patient([NAN, 80, NAN, NAN, 90])
    f = build_features(df, ["HR"], windows=[3])
    assert f["HR_observed"].tolist() == [0, 1, 0, 0, 1]
    assert np.allclose(f["HR_hours_since"], [NAN, 0, 1, 2, 0], equal_nan=True)


def test_hours_since_measured_directly():
    df = one_patient([5, NAN, NAN, 7])
    assert np.allclose(hours_since_measured(df, "HR"), [0, 1, 2, 0])


def test_missingness_indicators_can_be_switched_off():
    df = one_patient([1, NAN, 3])
    with_ind = build_features(df, ["HR"], windows=[3], missingness_indicators=True)
    without = build_features(df, ["HR"], windows=[3], missingness_indicators=False)
    assert {"HR_observed", "HR_hours_since"} <= set(with_ind.columns)
    assert not {"HR_observed", "HR_hours_since"} & set(without.columns)
    # Every other column is identical either way.
    assert_frame_equal(with_ind[without.columns], without)


# ---------- the key leakage test ----------

def test_features_at_t_ignore_the_future():
    df = random_cohort()
    cutoff = 12
    cols = ["HR", "Temp"]

    full = build_features(df, cols)
    full_before = full[full["ICULOS"] <= cutoff].reset_index(drop=True)

    # (a) Delete everything after the cutoff.
    truncated = build_features(df[df["ICULOS"] <= cutoff], cols)
    assert_frame_equal(full_before, truncated.reset_index(drop=True))

    # (b) Overwrite everything after the cutoff with junk (values and gaps).
    rng = np.random.default_rng(123)
    junk = df.copy()
    future = junk["ICULOS"] > cutoff
    for col in cols:
        junk.loc[future, col] = np.where(
            rng.random(future.sum()) < 0.3, np.nan, rng.normal(1000, 50, future.sum())
        )
    perturbed = build_features(junk, cols)
    perturbed_before = perturbed[perturbed["ICULOS"] <= cutoff].reset_index(drop=True)
    assert_frame_equal(full_before, perturbed_before)


# ---------- input checks ----------

def test_non_hourly_rows_raise_unless_check_is_off():
    df = pd.DataFrame({"patient_id": "p1", "ICULOS": [1, 2, 4], "HR": [1, 2, 3]})
    with pytest.raises(ValueError, match="hourly"):
        build_features(df, ["HR"], windows=[3])
    out = build_features(df, ["HR"], windows=[3], require_hourly=False)
    assert len(out) == 3


def test_duplicate_patient_hours_raise():
    df = pd.DataFrame({"patient_id": "p1", "ICULOS": [1, 2, 2], "HR": [1, 2, 3]})
    with pytest.raises(ValueError, match="duplicate"):
        build_features(df, ["HR"], windows=[3])


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Temp"):
        build_features(one_patient([1, 2, 3]), ["HR", "Temp"])


@pytest.mark.parametrize("windows", [[0], [-3]])
def test_bad_windows_raise(windows):
    with pytest.raises(ValueError):
        build_features(one_patient([1, 2, 3]), ["HR"], windows=windows)


def test_empty_value_columns_raise():
    with pytest.raises(ValueError):
        build_features(one_patient([1, 2, 3]), [])


def test_unsorted_input_gives_sorted_output_with_same_values():
    df = random_cohort()
    shuffled = df.sample(frac=1, random_state=7)
    a = build_features(df, ["HR", "Temp"])
    b = build_features(shuffled, ["HR", "Temp"])
    assert_frame_equal(a, b)
    keys = a[["patient_id", "ICULOS"]]
    assert keys.equals(keys.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True))


# ---------- medians fit on training data only ----------

def test_medians_come_from_training_data_only():
    train = pd.DataFrame(
        {"patient_id": "t", "ICULOS": [1, 2, 3], "HR_ffill": [60.0, 80.0, 100.0]}
    )
    test = pd.DataFrame(
        {"patient_id": "x", "ICULOS": [1, 2], "HR_ffill": [NAN, 200.0]}
    )
    medians = fit_medians(train)
    assert list(medians.index) == ["HR_ffill"]  # id and time columns excluded
    assert medians["HR_ffill"] == 80.0

    filled = apply_medians(test, medians)
    assert filled["HR_ffill"].tolist() == [80.0, 200.0]  # train median, not test's
    assert filled["ICULOS"].tolist() == [1, 2]
    assert np.isnan(test["HR_ffill"].iloc[0])  # input not modified
