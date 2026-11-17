"""Unit tests for the error-mode analysis.

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

from sepsis_icu.errors import (
    REVIEW_COLUMNS,
    add_subgroups,
    build_review_table,
    compare_profiles,
    false_alarm_profile,
    false_alarm_table,
    label_noise_candidates,
    miss_profile,
    patient_profiles,
    patient_table,
    shift_comparison,
    subgroup_metrics,
    update_review_file,
)
from sepsis_icu.leadtime import patient_lead_times
from sepsis_icu.metrics import METRIC_NAMES

NAN = np.nan


def rows(pid, labels, scores, **extra):
    """Patient-hours for one patient. Extra keyword columns are scalars or per-hour lists."""
    n = len(labels)
    d = pd.DataFrame(
        {"patient_id": pid, "ICULOS": np.arange(1, n + 1), "SepsisLabel": labels, "score": scores}
    )
    for name, value in extra.items():
        d[name] = value
    return d


# ---------- patient_table ----------

def table_cohort():
    return pd.concat(
        [
            rows("p1", [0, 0, 0, 0], [0.1] * 4, Age=70.0, Gender=1, Unit1=1, Unit2=0,
                 HR=[80, NAN, NAN, 80], Temp=[NAN] * 4),
            rows("p2", [0, 0, 1, 1], [0.1] * 4, Age=35.0, Gender=0, Unit1=0, Unit2=1,
                 HR=[80] * 4, Temp=[37.0] * 4),
            rows("p3", [0, 0, 0], [0.1] * 3, Age=NAN, Gender=NAN, Unit1=NAN, Unit2=NAN,
                 HR=[NAN] * 3, Temp=[37.0] * 3),
        ],
        ignore_index=True,
    )


def test_patient_table_values():
    pt = patient_table(table_cohort(), ["HR", "Temp"])
    assert list(pt.columns) == [
        "patient_id", "age", "sex", "unit", "stay_hours", "missing_fraction", "ever_septic"
    ]
    assert pt["patient_id"].tolist() == ["p1", "p2", "p3"]
    assert pt["age"].tolist()[:2] == [70.0, 35.0] and np.isnan(pt["age"].iloc[2])
    assert pt["sex"].tolist() == ["male", "female", "unknown"]
    assert pt["unit"].tolist() == ["MICU", "SICU", "unknown"]
    assert pt["stay_hours"].tolist() == [4, 4, 3]
    assert pt["missing_fraction"].tolist() == pytest.approx([0.75, 0.0, 0.5])
    assert pt["ever_septic"].tolist() == [False, True, False]


def test_patient_table_without_static_columns_gives_unknowns():
    pt = patient_table(table_cohort().drop(columns=["Gender", "Unit1", "Unit2"]), ["HR", "Temp"])
    assert (pt["sex"] == "unknown").all() and (pt["unit"] == "unknown").all()


def test_patient_table_rejects_bad_input():
    with pytest.raises(ValueError, match="Lactate"):
        patient_table(table_cohort(), ["HR", "Lactate"])
    with pytest.raises(ValueError, match="empty"):
        patient_table(table_cohort(), [])


# ---------- add_subgroups ----------

def test_age_and_stay_groups_at_every_boundary():
    pt = pd.DataFrame(
        {
            "age": [39, 40, 64, 65, 79, 80, NAN],
            "stay_hours": [23, 24, 71, 72, 167, 168, 10],
            "missing_fraction": np.linspace(0, 0.6, 7),
        }
    )
    out = add_subgroups(pt)
    assert out["age_group"].tolist() == ["<40", "40-64", "40-64", "65-79", "65-79", "80+", "unknown"]
    assert out["stay_group"].tolist() == ["<24h", "24-71h", "24-71h", "72-167h", "72-167h", "168h+", "<24h"]
    assert "age_group" not in pt.columns   # the input is not modified


def test_missing_data_tertiles():
    pt = pd.DataFrame(
        {
            "age": 50.0,
            "stay_hours": 30,
            "missing_fraction": [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6],
        }
    )
    out = add_subgroups(pt)
    by_value = dict(zip(pt["missing_fraction"], out["missing_group"]))
    assert {v for v, g in by_value.items() if g == "low"} == {0.1, 0.2, 0.3}
    assert {v for v, g in by_value.items() if g == "medium"} == {0.4, 0.5, 0.6}
    assert {v for v, g in by_value.items() if g == "high"} == {0.7, 0.8, 0.9}


def test_tied_missing_fractions_still_split_into_three_equal_groups():
    pt = pd.DataFrame({"age": 50.0, "stay_hours": 30, "missing_fraction": [0.5] * 6})
    assert add_subgroups(pt)["missing_group"].value_counts().to_dict() == {"low": 2, "medium": 2, "high": 2}


def test_missing_group_is_unknown_when_there_is_too_little_data():
    pt = pd.DataFrame({"age": [50.0, 60.0], "stay_hours": [30, 40], "missing_fraction": [0.1, 0.2]})
    assert (add_subgroups(pt)["missing_group"] == "unknown").all()
    pt = pd.DataFrame({"age": [50.0] * 4, "stay_hours": [30] * 4, "missing_fraction": [0.1, 0.2, NAN, 0.4]})
    assert add_subgroups(pt)["missing_group"].tolist()[2] == "unknown"


def test_add_subgroups_requires_the_columns():
    with pytest.raises(ValueError):
        add_subgroups(pd.DataFrame({"age": [1.0]}))


# ---------- subgroup_metrics ----------

def subgroup_cohort():
    df = pd.concat(
        [
            rows("s1", [0, 0, 1, 1], [0.1, 0.1, 0.9, 0.2]),
            rows("c1", [0, 0, 0, 0], [0.1, 0.6, 0.1, 0.1]),
            rows("s2", [0, 1, 1, 1], [0.1, 0.9, 0.9, 0.9]),
            rows("c2", [0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]),
        ],
        ignore_index=True,
    )
    patients = pd.DataFrame(
        {
            "patient_id": ["s1", "c1", "s2", "c2"],
            "grp": ["G", "G", "H", "H"],
            "ever_septic": [True, False, True, False],
        }
    )
    return df, patients


def test_subgroup_metrics_known_values():
    df, patients = subgroup_cohort()
    t = subgroup_metrics(df, patients, "grp", "score", threshold=0.5).set_index("group")
    assert list(t.index) == ["G", "H"]
    assert t["variable"].tolist() == ["grp", "grp"]
    assert t["n_patients"].tolist() == [2, 2]
    assert t["n_septic_patients"].tolist() == [1, 1]
    assert t["n_hours"].tolist() == [8, 8]
    assert t["prevalence"].tolist() == pytest.approx([0.25, 0.375])
    # Group G: alerts at s1 hour 3 (correct) and c1 hour 2 (false).
    assert t.loc["G", "sensitivity"] == pytest.approx(0.5)
    assert t.loc["G", "ppv"] == pytest.approx(0.5)
    assert t.loc["G", "alerts_per_patient_day"] == pytest.approx(2 / (8 / 24))
    # Group H: three alerts, all correct.
    assert t.loc["H", "sensitivity"] == pytest.approx(1.0)
    assert t.loc["H", "ppv"] == pytest.approx(1.0)
    assert t.loc["H", "alerts_per_patient_day"] == pytest.approx(3 / (8 / 24))
    assert set(METRIC_NAMES) <= set(t.columns)


def test_subgroup_metrics_low_n_flag():
    df, patients = subgroup_cohort()
    assert subgroup_metrics(df, patients, "grp", "score", 0.5)["low_n"].all()
    assert not subgroup_metrics(df, patients, "grp", "score", 0.5, min_septic_patients=1)["low_n"].any()


def test_subgroup_metrics_patient_level_sensitivity():
    df, patients = subgroup_cohort()
    lead = patient_lead_times(df, "score", 0.5)
    t = subgroup_metrics(df, patients, "grp", "score", 0.5, lead_table=lead).set_index("group")
    assert t["patient_sensitivity"].tolist() == [1.0, 1.0]
    assert "patient_sensitivity" not in subgroup_metrics(df, patients, "grp", "score", 0.5).columns

    # Add a septic patient in group G who is never alerted.
    df2 = pd.concat([df, rows("s3", [0, 0, 1, 1], [0.1] * 4)], ignore_index=True)
    patients2 = pd.concat(
        [patients, pd.DataFrame({"patient_id": ["s3"], "grp": ["G"], "ever_septic": [True]})],
        ignore_index=True,
    )
    lead2 = patient_lead_times(df2, "score", 0.5)
    t2 = subgroup_metrics(df2, patients2, "grp", "score", 0.5, lead_table=lead2).set_index("group")
    assert t2.loc["G", "patient_sensitivity"] == pytest.approx(0.5)
    assert t2.loc["H", "patient_sensitivity"] == pytest.approx(1.0)


def test_patients_missing_from_the_table_form_an_unknown_group():
    df, patients = subgroup_cohort()
    t = subgroup_metrics(df, patients[patients["patient_id"] != "c2"], "grp", "score", 0.5)
    assert set(t["group"]) == {"G", "H", "unknown"}
    assert t.set_index("group").loc["unknown", "n_patients"] == 1


def test_subgroup_metrics_requires_the_columns():
    df, patients = subgroup_cohort()
    with pytest.raises(ValueError):
        subgroup_metrics(df, patients, "nope", "score", 0.5)


# ---------- false_alarm_table ----------

def test_false_alarm_table_counts_only_never_septic_patients():
    df = pd.concat(
        [
            rows("c1", [0] * 4, [0.1, 0.6, 0.7, 0.1]),   # MICU, 2 alert hours
            rows("c2", [0] * 4, [0.1] * 4),              # MICU, none
            rows("c3", [0] * 4, [0.1, 0.9, 0.1, 0.1]),   # SICU, 1 alert hour
            rows("s", [0, 0, 1, 1], [0.9] * 4),          # septic: must be ignored
        ],
        ignore_index=True,
    )
    patients = pd.DataFrame(
        {
            "patient_id": ["c1", "c2", "c3", "s"],
            "unit": ["MICU", "MICU", "SICU", "MICU"],
            "ever_septic": [False, False, False, True],
        }
    )
    t = false_alarm_table(df, patients, "score", 0.5).set_index("group")
    assert list(t.index) == ["MICU", "SICU", "all"]
    assert t.loc["MICU", "n_patients"] == 2 and t.loc["MICU", "n_hours"] == 8
    assert t.loc["MICU", "alert_hours"] == 2
    assert t.loc["MICU", "alerts_per_patient_day"] == pytest.approx(6.0)
    assert t.loc["MICU", "pct_patients_with_alert"] == pytest.approx(50.0)
    assert t.loc["SICU", "alerts_per_patient_day"] == pytest.approx(6.0)
    assert t.loc["SICU", "pct_patients_with_alert"] == pytest.approx(100.0)
    assert t.loc["all", "n_patients"] == 3 and t.loc["all", "alert_hours"] == 3
    assert t.loc["all", "pct_patients_with_alert"] == pytest.approx(200 / 3)


def test_false_alarm_table_with_no_never_septic_patients():
    df = rows("s", [0, 1], [0.9, 0.9])
    patients = pd.DataFrame({"patient_id": ["s"], "unit": ["MICU"], "ever_septic": [True]})
    t = false_alarm_table(df, patients, "score", 0.5)
    assert t["group"].tolist() == ["all"]
    assert t["n_patients"].iloc[0] == 0 and np.isnan(t["alerts_per_patient_day"].iloc[0])


# ---------- patient_profiles and compare_profiles ----------

def test_patient_profiles_medians_cutoffs_and_observed_fraction():
    df = pd.concat(
        [
            rows("a", [0] * 4, [0.1] * 4, HR=[10, 20, 30, 40], Temp=[NAN, 37.0, NAN, 38.0]),
            rows("b", [0] * 4, [0.1] * 4, HR=[1, 2, 3, 4], Temp=[36.0] * 4),
        ],
        ignore_index=True,
    )
    p = patient_profiles(df, ["HR", "Temp"])
    assert p.loc["a", "HR"] == 25 and p.loc["a", "Temp"] == 37.5

    cut = patient_profiles(df, ["HR", "Temp"], up_to_hour={"a": 2})
    assert cut.loc["a", "HR"] == 15 and cut.loc["a", "Temp"] == 37.0   # hours 1-2 only
    assert cut.loc["b", "HR"] == 2.5                                    # not in the mapping: all rows

    frac = patient_profiles(df, ["HR", "Temp"], observed_fraction=True)
    assert frac.loc["a", "Temp_observed_frac"] == 0.5 and frac.loc["a", "HR_observed_frac"] == 1.0


def test_compare_profiles_values_and_ordering():
    profiles = pd.DataFrame(
        {
            "x": [1, 2, 3, 3, 4, 5],
            "y": [10, 10, 10, 10, 10, 10],
            "z": [1, 2, 3, 1.5, 2.5, 3.5],
        },
        index=["a1", "a2", "a3", "b1", "b2", "b3"],
    )
    t = compare_profiles(profiles, ["a1", "a2", "a3"], ["b1", "b2", "b3"], "a", "b")
    assert list(t.columns) == ["feature", "n_a", "n_b", "median_a", "median_b", "standardized_difference"]
    assert t["feature"].tolist() == ["x", "z", "y"]   # largest difference first, undefined last
    x = t.set_index("feature")
    assert x.loc["x", "standardized_difference"] == pytest.approx(-2.0)
    assert x.loc["z", "standardized_difference"] == pytest.approx(-0.5)
    assert np.isnan(x.loc["y", "standardized_difference"])   # no spread
    assert x.loc["x", "median_a"] == 2 and x.loc["x", "median_b"] == 4


def test_compare_profiles_with_an_empty_group():
    profiles = pd.DataFrame({"x": [1.0, 2.0]}, index=["a1", "a2"])
    t = compare_profiles(profiles, ["a1", "a2"], [], "a", "b")
    assert t["n_b"].iloc[0] == 0
    assert np.isnan(t["median_b"].iloc[0]) and np.isnan(t["standardized_difference"].iloc[0])


# ---------- false_alarm_profile ----------

def false_alarm_cohort():
    frames, ids, septic = [], [], []
    for i in range(40):
        hr = 100.0 if i >= 38 else 60.0 + (i % 3)
        frames.append(rows(f"c{i:02d}", [0] * 3, [i / 100] * 3, HR=hr))
        ids.append(f"c{i:02d}")
        septic.append(False)
    for i in range(3):   # septic patients with extreme values must be left out
        frames.append(rows(f"s{i}", [0, 1, 1], [0.99] * 3, HR=300.0))
        ids.append(f"s{i}")
        septic.append(True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame({"patient_id": ids, "ever_septic": septic})


def test_false_alarm_profile_compares_top_scorers_with_the_rest():
    df, patients = false_alarm_cohort()
    t = false_alarm_profile(df, patients, "score", ["HR"], top_fraction=0.05).iloc[0]
    assert t["n_high_scoring"] == 2 and t["n_other_non_septic"] == 38   # septic patients excluded
    assert t["median_high_scoring"] == 100.0
    assert 60 <= t["median_other_non_septic"] <= 62
    assert t["standardized_difference"] > 5


def test_false_alarm_profile_rejects_bad_fraction():
    df, patients = false_alarm_cohort()
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            false_alarm_profile(df, patients, "score", ["HR"], top_fraction=bad)


# ---------- miss_profile ----------

def miss_cohort():
    frames = []
    for k in range(3):
        hours = np.arange(1, 31)
        labels = (hours >= 13).astype(int)                       # onset at hour 19
        hr_detected = np.where(hours <= 19, 58.0 + 2 * k, 200.0)
        hr_missed = np.where(hours <= 19, 88.0 + 2 * k, 200.0)
        frames.append(rows(f"det{k}", labels, np.where(hours >= 10, 0.9, 0.1), HR=hr_detected))
        frames.append(rows(f"mis{k}", labels, [0.1] * 30, HR=hr_missed))
    # Label on from the first row: onset is left-censored and must be left out.
    frames.append(rows("lc", [1] * 30, [0.9] * 30, HR=500.0))
    return pd.concat(frames, ignore_index=True)


def test_miss_profile_uses_only_hours_up_to_onset_and_skips_left_censored():
    df = miss_cohort()
    lead = patient_lead_times(df, "score", 0.5)
    t = miss_profile(df, lead, ["HR"]).iloc[0]
    assert t["n_missed"] == 3 and t["n_detected"] == 3
    assert t["median_missed"] == 90 and t["median_detected"] == 60   # the 200s after onset are ignored
    assert t["standardized_difference"] == pytest.approx(15.0)


# ---------- label_noise_candidates ----------

def noise_cohort():
    hours = np.arange(1, 31)

    def scores_for(alert_hours):
        return np.where(np.isin(hours, list(alert_hours)), 0.9, 0.1)

    off_labels = np.zeros(30, dtype=int)
    off_labels[10:13] = 1   # hours 11-13, then back to 0
    return pd.concat(
        [
            rows("lc", [1] * 30, scores_for([])),                                        # label on at the first row
            rows("off", off_labels, scores_for([])),                                     # label turns off
            rows("late", (hours >= 28).astype(int), scores_for([])),                     # onset 34, record ends at 30
            rows("ok", (hours >= 13).astype(int), scores_for(range(5, 31))),             # septic, long alert run: fine
            rows("run12", [0] * 30, scores_for(range(5, 17))),                           # 12 alert hours in a row
            rows("run11", [0] * 30, scores_for(range(5, 16))),                           # 11
            rows("split", [0] * 30, scores_for(list(range(1, 7)) + list(range(8, 14)))), # 6 + 6, not in a row
        ],
        ignore_index=True,
    )


def test_label_noise_candidates():
    df = noise_cohort()
    lead = patient_lead_times(df, "score", 0.5)
    t = label_noise_candidates(df, "score", 0.5, lead)
    found = {(r.patient_id, r.flag): r.detail for r in t.itertuples()}
    assert found == {
        ("lc", "left_censored_onset"): 7.0,
        ("off", "label_turns_off"): 11.0,
        ("late", "onset_after_record_end"): 4.0,
        ("run12", "sustained_alerts_never_septic"): 12.0,
    }
    assert list(t.columns) == ["patient_id", "flag", "detail"]


def test_shorter_minimum_run_flags_more_never_septic_patients():
    df = noise_cohort()
    lead = patient_lead_times(df, "score", 0.5)
    t = label_noise_candidates(df, "score", 0.5, lead, min_alert_run=6)
    runs = t[t["flag"] == "sustained_alerts_never_septic"].set_index("patient_id")["detail"].to_dict()
    assert runs == {"run12": 12.0, "run11": 11.0, "split": 6.0}


def test_label_noise_with_nothing_to_flag_is_empty():
    df = rows("ok", [0] * 5 + [1] * 25, [0.1] * 30)
    lead = patient_lead_times(df, "score", 0.5)
    assert len(label_noise_candidates(df, "score", 0.5, lead)) == 0


# ---------- build_review_table and update_review_file ----------

def review_cohort():
    frames = []
    for i in range(12):                          # never septic, peak score i/20
        frames.append(rows(f"c{i:02d}", [0] * 20, [i / 20] * 20))
    for i in range(6):                           # septic, never alerted
        frames.append(rows(f"n{i}", [0] * 4 + [1] * 16, [0.1] * 20))
    for i in range(6):                           # septic, first alert at hour 14 (onset is hour 11)
        frames.append(rows(f"a{i}", [0] * 4 + [1] * 16, [0.1] * 13 + [0.9] * 7))
    return pd.concat(frames, ignore_index=True)


def review(df, **kwargs):
    lead = patient_lead_times(df, "score", 0.5)
    return build_review_table(df, "score", 0.5, lead, **kwargs)


def test_review_table_samples_each_error_type():
    t = review(review_cohort(), per_type=3)
    assert list(t.columns) == REVIEW_COLUMNS
    by_type = {k: g["patient_id"].tolist() for k, g in t.groupby("error_type")}
    assert sorted(by_type["false_alarm"]) == ["c09", "c10", "c11"]        # highest peak scores
    assert len(by_type["never_alerted"]) == 3 and all(p.startswith("n") for p in by_type["never_alerted"])
    assert len(by_type["alerted_after_onset"]) == 3 and all(p.startswith("a") for p in by_type["alerted_after_onset"])
    assert sorted(by_type["label_noise"]) == ["c10", "c11"]               # 20-hour alert runs, never septic

    top = t[(t["patient_id"] == "c11") & (t["error_type"] == "false_alarm")].iloc[0]
    assert top["max_score"] == pytest.approx(0.55) and top["alert_hours"] == 20 and top["stay_hours"] == 20
    assert "sustained_alerts_never_septic" in t[t["error_type"] == "label_noise"]["reason"].iloc[0]
    for blank in ("reviewer", "date_reviewed", "label_questionable", "finding", "notes"):
        assert (t[blank] == "").all()


def test_review_sample_is_reproducible_and_depends_on_the_seed():
    df = review_cohort()
    assert review(df, per_type=3, seed=5).equals(review(df, per_type=3, seed=5))
    samples = {tuple(review(df, per_type=3, seed=s).query("error_type == 'never_alerted'")["patient_id"]) for s in range(10)}
    assert len(samples) > 1


def test_review_table_rejects_bad_per_type():
    with pytest.raises(ValueError):
        review(review_cohort(), per_type=0)


def test_update_review_file_keeps_existing_notes(tmp_path):
    df = review_cohort()
    table = review(df, per_type=3)
    path = tmp_path / "results" / "manual_error_review.csv"

    first = update_review_file(str(path), table)
    assert path.exists() and len(first) == len(table)

    # A reviewer fills in a finding by hand.
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    saved.loc[0, "finding"] = "label looks wrong, chart shows infection on day 2"
    saved.loc[0, "reviewer"] = "me"
    saved.to_csv(path, index=False)

    # Re-running the analysis proposes the same rows plus one new one.
    extra = table.iloc[[0]].copy()
    extra["patient_id"] = "zz_new"
    merged = update_review_file(str(path), pd.concat([table, extra], ignore_index=True))

    on_disk = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(merged) == len(table) + 1 and len(on_disk) == len(merged)
    assert on_disk.loc[0, "finding"] == "label looks wrong, chart shows infection on day 2"
    assert on_disk.loc[0, "reviewer"] == "me"
    assert not on_disk.duplicated(["patient_id", "error_type"]).any()
    assert "zz_new" in set(on_disk["patient_id"])


def test_update_review_file_requires_the_key_columns(tmp_path):
    with pytest.raises(ValueError):
        update_review_file(str(tmp_path / "x.csv"), pd.DataFrame({"patient_id": ["a"]}))


# ---------- shift_comparison ----------

def simulate(n=20000, seed=0, logit_shift=0.0, score_shift=0.0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    z = np.log(p / (1 - p)) + logit_shift
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return y, np.clip(p + score_shift, 0, 1)


def test_shift_comparison_table():
    y_a, s_a = simulate(seed=0)
    y_b, s_b = simulate(seed=1, logit_shift=1.0, score_shift=0.1)
    t = shift_comparison(y_a, s_a, y_b, s_b, threshold=0.7).set_index("metric")
    assert t.index.tolist() == list(METRIC_NAMES)
    assert list(t.columns) == ["hospital_a", "hospital_b", "difference"]
    assert (t["difference"] == t["hospital_b"] - t["hospital_a"]).all()
    assert t.loc["alerts_per_patient_day", "difference"] > 0       # the alert rate moved up
    assert t.loc["prevalence", "difference"] > 0.1                 # B is a sicker population


def test_shift_comparison_without_probabilities_has_no_calibration_rows():
    y_a, s_a = simulate(seed=0)
    y_b, s_b = simulate(seed=1)
    t = shift_comparison(y_a, s_a * 10, y_b, s_b * 10, threshold=7.0, is_probability=False).set_index("metric")
    assert t.loc[["brier", "calibration_slope", "ece"], "hospital_a"].isna().all()
    assert not np.isnan(t.loc["auroc", "difference"])
