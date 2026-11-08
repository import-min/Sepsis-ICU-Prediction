"""Unit tests for the metrics module.

Hand-worked examples are checked exactly. If scikit-learn happens to be
installed, AUROC, AUPRC and Brier are also cross-checked against it (skipped
otherwise; scikit-learn is not required by the repo).

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from sepsis_icu.metrics import (
    METRIC_NAMES,
    auprc,
    auroc,
    bootstrap_ci,
    brier,
    calibration_intercept_slope,
    ece,
    evaluate,
    operating_point,
    patient_level,
    prevalence,
    reliability_table,
    threshold_for_alert_rate,
)

NAN = float("nan")


# ---------- input checks ----------

@pytest.mark.parametrize(
    "y, s",
    [
        ([0, 1], [0.1]),              # length mismatch
        ([], []),                      # empty
        ([0, 2], [0.1, 0.2]),         # non-binary label
        ([0, 1], [0.1, float("nan")]),  # NaN score
    ],
)
def test_bad_inputs_raise(y, s):
    with pytest.raises(ValueError):
        auroc(y, s)


# ---------- discrimination ----------

def test_prevalence():
    assert prevalence([0, 0, 0, 1]) == 0.25


def test_auroc_known_example():
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auroc_perfect_reversed_and_tied():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5  # ties count half


def test_auroc_needs_both_classes():
    assert np.isnan(auroc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_auprc_known_example():
    assert auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(5 / 6)


def test_auprc_perfect_and_tied():
    assert auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auprc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)  # one tied threshold


def test_auprc_without_positives_is_nan():
    assert np.isnan(auprc([0, 0], [0.1, 0.2]))


def test_matches_scikit_learn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    s = np.round(np.clip(0.1 + 0.25 * y + rng.normal(0, 0.15, 3000), 0, 1), 2)  # many ties
    assert auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    assert auprc(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert brier(y, s) == pytest.approx(sk.brier_score_loss(y, s))


# ---------- calibration ----------

def test_brier_known_value_and_range_check():
    assert brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)
    with pytest.raises(ValueError):
        brier([1, 0], [1.2, 0.3])


def simulate(logit_of_truth, n=60000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    logit = np.log(p / (1 - p))
    true_p = 1 / (1 + np.exp(-logit_of_truth(logit)))
    y = (rng.random(n) < true_p).astype(int)
    return y, p


def test_calibration_slope_and_intercept_for_well_calibrated_predictions():
    y, p = simulate(lambda z: z)
    intercept, slope = calibration_intercept_slope(y, p)
    assert intercept == pytest.approx(0.0, abs=0.08)
    assert slope == pytest.approx(1.0, abs=0.08)


def test_overconfident_predictions_have_slope_below_one():
    y, p = simulate(lambda z: 0.5 * z)
    _, slope = calibration_intercept_slope(y, p)
    assert slope == pytest.approx(0.5, abs=0.08)


def test_predictions_that_are_too_low_have_positive_intercept():
    y, p = simulate(lambda z: z + 1.0)
    intercept, slope = calibration_intercept_slope(y, p)
    assert intercept == pytest.approx(1.0, abs=0.08)
    assert slope == pytest.approx(1.0, abs=0.08)


def test_calibration_fit_returns_nan_when_it_cannot_be_fit():
    i, s = calibration_intercept_slope([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3])  # no spread
    assert np.isnan(i) and np.isnan(s)
    i, s = calibration_intercept_slope([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])  # perfect separation
    assert np.isnan(i) and np.isnan(s)


def test_reliability_table_values():
    t = reliability_table([0, 1, 1, 1], [0.05, 0.05, 0.95, 0.95], n_bins=10)
    assert len(t) == 10
    assert list(t.columns) == ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"]
    assert t.loc[0, "n"] == 2
    assert t.loc[0, "observed_rate"] == 0.5
    assert t.loc[9, "n"] == 2
    assert t.loc[9, "observed_rate"] == 1.0
    assert t["n"].sum() == 4
    assert np.isnan(t.loc[4, "mean_predicted"])  # empty bin


def test_probability_of_exactly_one_goes_in_last_bin():
    t = reliability_table([1], [1.0], n_bins=10)
    assert t.loc[9, "n"] == 1


def test_ece_values():
    assert ece([0, 1, 1, 1], [0.05, 0.05, 0.95, 0.95]) == pytest.approx(0.25)
    assert ece([1, 0, 0, 0], [0.25] * 4) == pytest.approx(0.0)


# ---------- operating point ----------

def test_threshold_gives_exact_alert_count_without_ties():
    s = np.arange(2400) / 2400  # 2400 patient-hours = 100 patient-days
    thr = threshold_for_alert_rate(s, 3.0)  # 3/day -> 300 alerts
    assert thr == pytest.approx(2100 / 2400)
    assert np.sum(s >= thr) == 300


def test_threshold_with_ties_never_exceeds_the_target():
    s = np.array([1.0] * 100 + [0.0] * 2300)
    thr = threshold_for_alert_rate(s, 3.0)  # allowed 300; alerting on 0.0 would give 2400
    assert thr == 1.0
    assert np.sum(s >= thr) == 100


def test_threshold_is_infinite_when_no_alerts_are_possible():
    assert threshold_for_alert_rate(np.full(2400, 1.0), 3.0) == float("inf")  # all tied
    assert threshold_for_alert_rate(np.arange(100) / 100, 0.0) == float("inf")
    assert threshold_for_alert_rate(np.arange(24) / 24, 0.5) == float("inf")  # < 1 alert allowed


def test_threshold_allows_everything_when_target_is_high():
    s = np.arange(240) / 240
    assert threshold_for_alert_rate(s, 24.0) == s.min()


def test_threshold_is_the_lowest_one_that_meets_the_target():
    rng = np.random.default_rng(3)
    s = np.round(rng.random(4800), 1)  # many ties
    target = 2.0
    allowed = int(np.floor(target * s.size / 24))
    thr = threshold_for_alert_rate(s, target)
    assert np.sum(s >= thr) <= allowed
    lower = np.unique(s[s < thr])
    if lower.size:  # the next lower distinct score would break the target
        assert np.sum(s >= lower.max()) > allowed


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        threshold_for_alert_rate([], 1.0)
    with pytest.raises(ValueError):
        threshold_for_alert_rate([0.1, 0.2], -1.0)


def test_operating_point_known_example():
    y = [1, 1, 0, 0, 0, 1]
    s = [0.9, 0.4, 0.8, 0.2, 0.1, 0.7]
    op = operating_point(y, s, 0.5)  # alerts at 0.9 (TP), 0.8 (FP), 0.7 (TP)
    assert op["sensitivity"] == pytest.approx(2 / 3)
    assert op["ppv"] == pytest.approx(2 / 3)
    assert op["alerts_per_patient_day"] == pytest.approx(3 / (6 / 24))


def test_operating_point_threshold_is_inclusive_and_handles_no_alerts():
    assert operating_point([1, 0], [0.5, 0.1], 0.5)["sensitivity"] == 1.0
    none = operating_point([1, 0], [0.5, 0.1], float("inf"))
    assert none["sensitivity"] == 0.0
    assert np.isnan(none["ppv"])
    assert none["alerts_per_patient_day"] == 0.0
    assert np.isnan(operating_point([0, 0], [0.5, 0.1], 0.2)["sensitivity"])


# ---------- patient-level view ----------

def test_patient_level_collapses_rows():
    ever, highest = patient_level(
        ["a", "a", "b", "b", "c"], [0, 1, 0, 0, 0], [0.1, 0.9, 0.3, 0.2, 0.5]
    )
    assert ever.tolist() == [1, 0, 0]
    assert highest.tolist() == [0.9, 0.3, 0.5]


# ---------- evaluate ----------

def test_evaluate_returns_every_metric_for_probabilities():
    y, p = simulate(lambda z: z, n=5000)
    out = evaluate(y, p, threshold=0.7)
    assert tuple(out) == METRIC_NAMES
    assert not any(np.isnan(v) for v in out.values())


def test_evaluate_without_threshold_leaves_operating_point_empty():
    y, p = simulate(lambda z: z, n=2000)
    out = evaluate(y, p)
    assert np.isnan(out["sensitivity"]) and np.isnan(out["ppv"])
    assert not np.isnan(out["auroc"])


def test_evaluate_skips_calibration_for_non_probability_scores():
    out = evaluate([0, 0, 1, 1, 1], [0, 1, 2, 3, 4], threshold=2, is_probability=False)
    for name in ("brier", "calibration_intercept", "calibration_slope", "ece"):
        assert np.isnan(out[name])
    assert out["auroc"] == 1.0
    assert out["sensitivity"] == 1.0


def test_evaluate_rejects_non_probabilities_by_default():
    with pytest.raises(ValueError):
        evaluate([0, 1], [0, 5])


# ---------- patient-level bootstrap ----------

def cohort(n_patients=100, seed=0):
    rng = np.random.default_rng(seed)
    ids, y, s = [], [], []
    for i in range(n_patients):
        n = int(rng.integers(8, 20))
        septic = rng.random() < 0.3
        ids += [f"p{i}"] * n
        y += [int(septic)] * n
        s += list(np.clip(rng.normal(0.6 if septic else 0.35, 0.2, n), 0, 1))
    return np.array(ids), np.array(y), np.array(s)


def test_bootstrap_table_shape_and_estimate():
    ids, y, s = cohort()
    t = bootstrap_ci(ids, y, s, lambda a, b: evaluate(a, b, threshold=0.5), n_resamples=60)
    assert list(t.columns) == ["metric", "estimate", "ci_lower", "ci_upper", "n_valid"]
    assert t["metric"].tolist() == list(METRIC_NAMES)
    auroc_row = t[t["metric"] == "auroc"].iloc[0]
    assert auroc_row["estimate"] == pytest.approx(auroc(y, s))
    assert auroc_row["ci_lower"] <= auroc_row["estimate"] <= auroc_row["ci_upper"]
    assert auroc_row["n_valid"] == 60


def test_bootstrap_is_reproducible_and_seed_dependent():
    ids, y, s = cohort()
    fn = lambda a, b: {"auroc": auroc(a, b)}
    t1 = bootstrap_ci(ids, y, s, fn, n_resamples=50, seed=7)
    t2 = bootstrap_ci(ids, y, s, fn, n_resamples=50, seed=7)
    t3 = bootstrap_ci(ids, y, s, fn, n_resamples=50, seed=8)
    assert t1.equals(t2)
    assert not t1.equals(t3)


def test_bootstrap_resamples_whole_patients_not_rows():
    lengths = np.arange(5, 35)  # 30 patients with different stay lengths
    ids = np.repeat(np.arange(30), lengths)
    scores = ids.astype(float)  # the score encodes which patient a row came from
    y = (ids % 2).astype(int)

    def all_rows_of_each_patient_together(_y, s):
        counts = np.bincount(s.astype(int), minlength=30)
        return {"whole_patients": float(np.all(counts % lengths == 0))}

    t = bootstrap_ci(ids, y, scores, all_rows_of_each_patient_together, n_resamples=50)
    row = t.iloc[0]
    assert row["ci_lower"] == 1.0 and row["ci_upper"] == 1.0 and row["n_valid"] == 50


def test_bootstrap_draws_as_many_patients_as_the_original_sample():
    ids = np.repeat(np.arange(20), 10)  # 20 patients x 10 hours
    y = np.zeros(200, dtype=int)
    s = np.zeros(200)
    t = bootstrap_ci(ids, y, s, lambda a, b: {"n_rows": float(len(a))}, n_resamples=30)
    assert t.iloc[0]["ci_lower"] == 200 and t.iloc[0]["ci_upper"] == 200


def test_bootstrap_perfect_separation_gives_degenerate_interval():
    ids = np.repeat(np.arange(40), 5)
    y = (ids % 2).astype(int)
    s = y.astype(float)
    t = bootstrap_ci(ids, y, s, lambda a, b: {"auroc": auroc(a, b)}, n_resamples=40)
    assert t.iloc[0]["ci_lower"] == 1.0 and t.iloc[0]["ci_upper"] == 1.0


def test_bootstrap_counts_resamples_where_metric_is_undefined():
    # One septic patient out of five: some resamples contain no septic patient.
    ids = np.repeat(np.arange(5), 6)
    y = (ids == 0).astype(int)
    s = np.random.default_rng(0).random(30)
    t = bootstrap_ci(ids, y, s, lambda a, b: {"auroc": auroc(a, b)}, n_resamples=200)
    n_valid = int(t.iloc[0]["n_valid"])
    assert 0 < n_valid < 200


def test_wider_confidence_level_gives_wider_interval():
    ids, y, s = cohort()
    fn = lambda a, b: {"auroc": auroc(a, b)}
    narrow = bootstrap_ci(ids, y, s, fn, n_resamples=100, ci_level=0.5).iloc[0]
    wide = bootstrap_ci(ids, y, s, fn, n_resamples=100, ci_level=0.95).iloc[0]
    assert (wide["ci_upper"] - wide["ci_lower"]) > (narrow["ci_upper"] - narrow["ci_lower"])


def test_bootstrap_rejects_bad_arguments():
    ids, y, s = cohort(10)
    fn = lambda a, b: {"auroc": auroc(a, b)}
    with pytest.raises(ValueError):
        bootstrap_ci(ids[:-1], y, s, fn)
    with pytest.raises(ValueError):
        bootstrap_ci(ids, y, s, fn, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci(ids, y, s, fn, ci_level=1.0)
