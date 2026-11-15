"""Unit tests for recalibration.

Isotonic regression is also cross-checked against scikit-learn if it happens
to be installed (skipped otherwise).

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from sepsis_icu.metrics import auroc, calibration_intercept_slope, ece
from sepsis_icu.recalibration import (
    InterceptShift,
    IsotonicCalibrator,
    PlattScaler,
    compare_calibration,
    expit,
    logit,
    split_labeled_portion,
)
from sepsis_icu.splits import patient_outcomes


def simulate(truth_logit, n=60000, seed=0):
    """Predictions p and outcomes y whose true risk is expit(truth_logit(logit(p)))."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    true_p = expit(truth_logit(logit(p)))
    y = (rng.random(n) < true_p).astype(int)
    return p, y


# ---------- logit / expit ----------

def test_logit_and_expit_are_inverses():
    p = np.array([0.01, 0.2, 0.5, 0.9, 0.999])
    assert np.allclose(expit(logit(p)), p)


def test_logit_is_finite_at_zero_and_one():
    assert np.isfinite(logit([0.0, 1.0])).all()


def test_expit_does_not_overflow():
    out = expit([-1e6, 1e6])
    assert np.isfinite(out).all()
    assert np.allclose(out, [0.0, 1.0], atol=1e-12)


# ---------- Platt scaling ----------

def test_platt_recovers_known_miscalibration():
    p, y = simulate(lambda z: 0.5 * z + 0.3, seed=0)
    platt = PlattScaler().fit(p, y)
    assert platt.slope_ == pytest.approx(0.5, abs=0.06)
    assert platt.intercept_ == pytest.approx(0.3, abs=0.08)


def test_platt_fixes_calibration_on_new_data():
    p_dev, y_dev = simulate(lambda z: 0.5 * z + 0.3, seed=0)
    p_new, y_new = simulate(lambda z: 0.5 * z + 0.3, seed=1)
    platt = PlattScaler().fit(p_dev, y_dev)
    before = calibration_intercept_slope(y_new, p_new)
    after = calibration_intercept_slope(y_new, platt.transform(p_new))
    assert abs(before[1] - 1) > 0.4          # slope was far from 1
    assert after[1] == pytest.approx(1.0, abs=0.08)
    assert after[0] == pytest.approx(0.0, abs=0.08)
    assert ece(y_new, platt.transform(p_new)) < ece(y_new, p_new)


def test_platt_leaves_well_calibrated_predictions_alone():
    p, y = simulate(lambda z: z)
    platt = PlattScaler().fit(p, y)
    assert platt.slope_ == pytest.approx(1.0, abs=0.08)
    assert platt.intercept_ == pytest.approx(0.0, abs=0.08)


def test_platt_does_not_change_auroc():
    p, y = simulate(lambda z: 0.5 * z + 0.3)
    platt = PlattScaler().fit(p, y)
    assert auroc(y, platt.transform(p)) == pytest.approx(auroc(y, p))


def test_platt_cannot_be_fit_without_spread_or_with_perfect_separation():
    with pytest.raises(ValueError, match="Platt"):
        PlattScaler().fit([0.3, 0.3, 0.3, 0.3], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="Platt"):
        PlattScaler().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])


# ---------- intercept-only update ----------

def test_intercept_shift_recovers_known_shift():
    p, y = simulate(lambda z: z + 1.0)
    update = InterceptShift().fit(p, y)
    assert update.shift_ == pytest.approx(1.0, abs=0.06)


def test_intercept_shift_makes_mean_prediction_equal_observed_rate():
    p, y = simulate(lambda z: z - 0.7, n=5000)
    update = InterceptShift().fit(p, y)
    assert update.transform(p).sum() == pytest.approx(y.sum(), rel=1e-6)


def test_intercept_shift_does_not_change_auroc():
    p, y = simulate(lambda z: z + 1.0)
    update = InterceptShift().fit(p, y)
    assert auroc(y, update.transform(p)) == pytest.approx(auroc(y, p))


def test_shift_is_near_zero_when_already_calibrated_in_the_large():
    p, y = simulate(lambda z: z)
    assert InterceptShift().fit(p, y).shift_ == pytest.approx(0.0, abs=0.06)


# ---------- isotonic ----------

def test_isotonic_known_example():
    iso = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    # 0.2 and 0.3 violate the order and pool to 0.5.
    assert iso.transform([0.1, 0.2, 0.25, 0.3, 0.4]).tolist() == pytest.approx([0, 0.5, 0.5, 0.5, 1])
    assert iso.transform([0.15]) == pytest.approx([0.25])  # linear between knots


def test_isotonic_pools_tied_predictions():
    iso = IsotonicCalibrator().fit([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0])
    assert iso.transform([0.0, 0.5, 1.0]).tolist() == pytest.approx([0.25, 0.25, 0.25])


def test_isotonic_output_is_monotone_and_clips_outside_the_fitted_range():
    p, y = simulate(lambda z: z, n=5000)
    iso = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0, 1, 500)
    out = iso.transform(grid)
    assert (np.diff(out) >= -1e-12).all()
    assert ((out >= 0) & (out <= 1)).all()
    assert iso.transform([0.0])[0] == iso.transform([p.min()])[0]
    assert iso.transform([1.0])[0] == iso.transform([p.max()])[0]


def test_isotonic_fixes_a_nonlinear_distortion():
    # True risk is p squared, which no straight line in logit space fixes well.
    p_dev, y_dev = simulate(lambda z: np.log(expit(z) ** 2 / (1 - expit(z) ** 2)), seed=0)
    p_new, y_new = simulate(lambda z: np.log(expit(z) ** 2 / (1 - expit(z) ** 2)), seed=1)
    iso = IsotonicCalibrator().fit(p_dev, y_dev)
    before, after = ece(y_new, p_new), ece(y_new, iso.transform(p_new))
    assert before > 0.1
    assert after < 0.03


def test_isotonic_matches_scikit_learn_when_available():
    sk = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(2)
    p = np.round(rng.random(3000), 2)  # many ties
    y = (rng.random(3000) < p ** 2).astype(int)
    ours = IsotonicCalibrator().fit(p, y)
    theirs = sk.IsotonicRegression(out_of_bounds="clip").fit(p, y)
    grid = np.linspace(0, 1, 301)
    assert np.allclose(ours.transform(grid), theirs.predict(grid), atol=1e-9)


# ---------- shared behavior ----------

@pytest.mark.parametrize("calibrator", [PlattScaler, InterceptShift, IsotonicCalibrator])
def test_transform_before_fit_raises(calibrator):
    with pytest.raises(RuntimeError, match="fit"):
        calibrator().transform([0.1, 0.2])


@pytest.mark.parametrize("calibrator", [PlattScaler, InterceptShift, IsotonicCalibrator])
def test_fit_validates_inputs(calibrator):
    with pytest.raises(ValueError):
        calibrator().fit([0.1, 0.2, 0.3], [0, 1])               # length mismatch
    with pytest.raises(ValueError):
        calibrator().fit([0.1, 1.2, 0.3], [0, 1, 0])            # not probabilities
    with pytest.raises(ValueError):
        calibrator().fit([0.1, 0.2, 0.3], [0, 2, 1])            # non-binary labels
    with pytest.raises(ValueError, match="both"):
        calibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])            # one outcome only
    with pytest.raises(ValueError):
        calibrator().fit([0.1, float("nan"), 0.3], [0, 1, 0])   # NaN


# ---------- split_labeled_portion ----------

def patient_rows(n_patients=400, rows_each=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.repeat([f"B_p{i:04d}" for i in range(n_patients)], rows_each)
    septic = rng.random(n_patients) < 0.25
    y = np.zeros(n_patients * rows_each, dtype=int)
    for i in np.flatnonzero(septic):
        y[i * rows_each + 3:(i + 1) * rows_each] = 1
    return ids, y


def test_labeled_portion_is_patient_level_stratified_and_disjoint():
    ids, y = patient_rows()
    labeled, evaluation = split_labeled_portion(ids, y, labeled_fraction=0.2)
    all_ids, ever = patient_outcomes(ids, y)
    assert not set(labeled) & set(evaluation)
    assert len(labeled) + len(evaluation) == len(all_ids)
    septic_ids = set(all_ids[ever])
    assert len(set(labeled) & septic_ids) == round(0.2 * len(septic_ids))


def test_labeled_portion_is_deterministic_and_seed_dependent():
    ids, y = patient_rows()
    a = split_labeled_portion(ids, y, 0.2, seed=1)
    b = split_labeled_portion(ids, y, 0.2, seed=1)
    c = split_labeled_portion(ids, y, 0.2, seed=2)
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.5])
def test_labeled_portion_rejects_bad_fraction(fraction):
    ids, y = patient_rows()
    with pytest.raises(ValueError):
        split_labeled_portion(ids, y, labeled_fraction=fraction)


def test_labeled_portion_rejects_a_fraction_that_empties_a_stratum():
    ids, y = patient_rows(n_patients=12)  # about 3 septic patients; 10% rounds to 0
    with pytest.raises(ValueError, match="labeled_fraction"):
        split_labeled_portion(ids, y, labeled_fraction=0.1)


# ---------- compare_calibration ----------

def test_compare_calibration_table():
    p, y = simulate(lambda z: z + 1.0, n=20000)
    shifted = InterceptShift().fit(p, y).transform(p)
    table = compare_calibration(y, {"uncalibrated": p, "intercept_update": shifted})
    assert table["variant"].tolist() == ["uncalibrated", "intercept_update"]
    assert list(table.columns) == [
        "variant", "auroc", "brier", "calibration_intercept",
        "calibration_slope", "ece", "mean_predicted", "observed_rate",
    ]
    raw, fixed = table.iloc[0], table.iloc[1]
    assert abs(fixed["calibration_intercept"]) < abs(raw["calibration_intercept"])
    assert fixed["mean_predicted"] == pytest.approx(fixed["observed_rate"], rel=1e-3)
    assert fixed["auroc"] == pytest.approx(raw["auroc"])


def test_compare_calibration_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compare_calibration([0, 1, 0], {"x": [0.1, 0.2]})


# ---------- the hospital B workflow from the study design ----------

def test_external_site_workflow_intercept_update_on_a_labeled_portion():
    rng = np.random.default_rng(7)

    # Development: predictions are well calibrated. Platt scaling is fit here.
    p_dev = rng.beta(1, 30, 40000)
    y_dev = (rng.random(40000) < expit(logit(p_dev))).astype(int)
    platt = PlattScaler().fit(p_dev, y_dev)

    # Hospital B: same score, but the true risk is higher (a shift of +1 in logit).
    n_patients, rows_each = 3000, 8
    ids = np.repeat([f"B_p{i:05d}" for i in range(n_patients)], rows_each)
    p_b = rng.beta(1, 30, n_patients * rows_each)
    y_b = (rng.random(p_b.size) < expit(logit(p_b) + 1.0)).astype(int)

    labeled_ids, eval_ids = split_labeled_portion(ids, y_b, labeled_fraction=0.2)
    labeled = np.isin(ids, labeled_ids)
    evaluation = np.isin(ids, eval_ids)
    assert not (labeled & evaluation).any()

    update = InterceptShift().fit(p_b[labeled], y_b[labeled])
    assert update.shift_ == pytest.approx(1.0, abs=0.25)

    table = compare_calibration(
        y_b[evaluation],
        {
            "uncalibrated": p_b[evaluation],
            "platt_from_development": platt.transform(p_b[evaluation]),
            "intercept_update_from_B": update.transform(p_b[evaluation]),
        },
    ).set_index("variant")

    intercept = table["calibration_intercept"]
    # Development-fitted Platt scaling cannot fix a shift it never saw ...
    assert abs(intercept["platt_from_development"]) > 0.6
    # ... but a small labeled portion of B can.
    assert abs(intercept["intercept_update_from_B"]) < 0.25
    assert abs(intercept["intercept_update_from_B"]) < abs(intercept["uncalibrated"])
    assert table.loc["intercept_update_from_B", "brier"] <= table.loc["uncalibrated", "brier"]
