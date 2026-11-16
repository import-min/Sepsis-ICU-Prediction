"""Unit tests for lead-time bins, matched controls, trajectories, and utility.

The utility score is checked three ways: hand-worked values, a slow row-by-row
reference written in this file, and the constants of the official 2019
Challenge scoring code.

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

from sepsis_icu.leadtime import (
    BIN_ORDER,
    DT_EARLY,
    DT_LATE,
    DT_OPTIMAL,
    MAX_U_TP,
    MIN_U_FN,
    U_FP,
    U_TN,
    bin_summary,
    detection_summary,
    match_controls,
    normalized_utility,
    patient_lead_times,
    prediction_utility,
    score_trajectories,
)
from sepsis_icu.onset import BIN_EARLY, BIN_INTERMEDIATE, BIN_LATE, BIN_MISSED


# ---------- a small cohort with known answers ----------

def patient(pid, n=30, label_from=None, alert_from=None, alert_score=0.9):
    """Hours 1..n. Label is 1 from `label_from`; score jumps to 0.9 from `alert_from`."""
    hours = np.arange(1, n + 1)
    label = (hours >= label_from).astype(int) if label_from else np.zeros(n, dtype=int)
    score = np.where(hours >= alert_from, alert_score, 0.1) if alert_from else np.full(n, 0.1)
    return pd.DataFrame({"patient_id": pid, "ICULOS": hours, "SepsisLabel": label, "score": score})


def cohort():
    # Label from hour 13 means onset at hour 19.
    return pd.concat(
        [
            patient("s1", label_from=13, alert_from=5),    # lead 14 -> early
            patient("s2", label_from=13, alert_from=10),   # lead 9  -> intermediate
            patient("s3", label_from=13, alert_from=16),   # lead 3  -> late
            patient("s4", label_from=13, alert_from=21),   # lead -2 -> after onset
            patient("s5", label_from=13),                  # never alerted
            patient("s6", label_from=1, alert_from=1),     # label on at first row: onset 7, left-censored
            patient("c1"),                                 # never septic, no alerts
            patient("c2", alert_from=3),                   # never septic, false alarm
        ],
        ignore_index=True,
    )


# ---------- patient_lead_times ----------

def test_lead_time_table_for_each_kind_of_patient():
    t = patient_lead_times(cohort(), "score", threshold=0.5).set_index("patient_id")
    assert list(t.index) == ["s1", "s2", "s3", "s4", "s5", "s6"]   # no controls, sorted
    assert t["onset_hour"].tolist() == [19, 19, 19, 19, 19, 7]
    assert t["lead_time_hours"].iloc[:4].tolist() == [14, 9, 3, -2]
    assert np.isnan(t.loc["s5", "lead_time_hours"]) and np.isnan(t.loc["s5", "first_alert_hour"])
    assert t["bin"].tolist() == [
        BIN_EARLY, BIN_INTERMEDIATE, BIN_LATE, BIN_MISSED, BIN_MISSED, BIN_INTERMEDIATE
    ]
    assert t["left_censored"].tolist() == [False, False, False, False, False, True]
    assert t["label_monotone"].all()


def test_threshold_is_inclusive():
    df = patient("s1", label_from=13, alert_from=5, alert_score=0.5)
    t = patient_lead_times(df, "score", threshold=0.5)
    assert t["first_alert_hour"].iloc[0] == 5


def test_missing_scores_never_alert():
    df = patient("s1", label_from=13, alert_from=15)
    df.loc[df["ICULOS"] == 15, "score"] = np.nan   # the model gave no score that hour
    t = patient_lead_times(df, "score", threshold=0.5)
    assert t["first_alert_hour"].iloc[0] == 16


def test_row_order_does_not_matter():
    a = patient_lead_times(cohort(), "score", 0.5)
    b = patient_lead_times(cohort().sample(frac=1, random_state=3), "score", 0.5)
    pd.testing.assert_frame_equal(a, b)


def test_no_septic_patients_gives_an_empty_table_with_columns():
    t = patient_lead_times(patient("c1"), "score", 0.5)
    assert len(t) == 0
    assert "lead_time_hours" in t.columns and "bin" in t.columns


def test_missing_column_raises():
    with pytest.raises(ValueError, match="score"):
        patient_lead_times(cohort().drop(columns=["score"]), "score", 0.5)


# ---------- bin_summary and detection_summary ----------

def test_bin_summary_counts_and_order_excluding_left_censored():
    t = patient_lead_times(cohort(), "score", 0.5)
    s = bin_summary(t, n_resamples=0)
    assert s["bin"].tolist() == list(BIN_ORDER)
    assert s["n_patients"].tolist() == [1, 1, 1, 2]          # s6 excluded
    assert s["fraction"].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.4])


def test_bin_summary_can_include_left_censored_patients():
    t = patient_lead_times(cohort(), "score", 0.5)
    s = bin_summary(t, exclude_left_censored=False, n_resamples=0)
    assert s["n_patients"].tolist() == [1, 2, 1, 2]
    assert s["fraction"].sum() == pytest.approx(1.0)


def test_bin_summary_always_lists_all_four_bins():
    t = pd.DataFrame({"left_censored": [False, False], "bin": [BIN_EARLY, BIN_EARLY]})
    s = bin_summary(t, n_resamples=0)
    assert s["n_patients"].tolist() == [2, 0, 0, 0]


def balanced_table(n_per_bin):
    bins = [b for b in BIN_ORDER for _ in range(n_per_bin)]
    return pd.DataFrame({"left_censored": [False] * len(bins), "bin": bins})


def test_bin_summary_confidence_intervals():
    s = bin_summary(balanced_table(50), n_resamples=500)
    assert (s["ci_lower"] <= s["fraction"]).all() and (s["fraction"] <= s["ci_upper"]).all()
    assert (s["ci_upper"] - s["ci_lower"]).between(0.05, 0.25).all()
    # Reproducible with the same seed; no intervals when resampling is switched off.
    pd.testing.assert_frame_equal(s, bin_summary(balanced_table(50), n_resamples=500))
    assert bin_summary(balanced_table(50), n_resamples=0)["ci_lower"].isna().all()


def test_more_patients_give_narrower_intervals():
    small = bin_summary(balanced_table(25), n_resamples=500)
    large = bin_summary(balanced_table(250), n_resamples=500)
    assert (large["ci_upper"] - large["ci_lower"]).mean() < (small["ci_upper"] - small["ci_lower"]).mean()


def test_bin_summary_with_no_patients_is_nan():
    empty = pd.DataFrame({"left_censored": pd.Series([], dtype=bool), "bin": pd.Series([], dtype=object)})
    s = bin_summary(empty)
    assert s["n_patients"].tolist() == [0, 0, 0, 0]
    assert s["fraction"].isna().all()


def test_bin_summary_rejects_bad_arguments():
    t = balanced_table(5)
    with pytest.raises(ValueError):
        bin_summary(t, ci_level=1.0)
    with pytest.raises(ValueError):
        bin_summary(t, n_resamples=-1)
    with pytest.raises(ValueError, match="unknown bin"):
        bin_summary(pd.DataFrame({"left_censored": [False], "bin": ["sometime"]}))


def test_detection_summary():
    t = patient_lead_times(cohort(), "score", 0.5)
    d = detection_summary(t)
    assert d["n_septic_patients"] == 5
    assert d["n_left_censored_excluded"] == 1
    assert d["n_alerted_before_or_at_onset"] == 3
    assert d["patient_sensitivity"] == pytest.approx(0.6)
    assert d["n_alerted_only_after_onset"] == 1
    assert d["n_never_alerted"] == 1
    assert d["median_lead_time_hours"] == 9      # median of 14, 9, 3

    d_all = detection_summary(t, exclude_left_censored=False)
    assert d_all["n_septic_patients"] == 6
    assert d_all["median_lead_time_hours"] == 7.5  # median of 14, 9, 3, 6


def test_detection_summary_with_no_patients():
    t = patient_lead_times(patient("c1"), "score", 0.5)
    d = detection_summary(t)
    assert d["n_septic_patients"] == 0
    assert np.isnan(d["patient_sensitivity"]) and np.isnan(d["median_lead_time_hours"])


# ---------- match_controls ----------

def stay(pid, last_hour, label_from=None):
    return patient(pid, n=last_hour, label_from=label_from)


def onsets(**pid_to_onset):
    return pd.DataFrame({"patient_id": list(pid_to_onset), "onset_hour": [float(v) for v in pid_to_onset.values()]})


def test_control_is_eligible_and_closest_in_stay_length():
    df = pd.concat(
        [
            stay("A", 20, label_from=4),   # onset 10, stays until hour 20
            stay("c1", 8),                 # left before onset: not eligible
            stay("c2", 18),                # gap 2
            stay("c3", 30),                # gap 10
            stay("c4", 21),                # gap 1 -> the match
        ],
        ignore_index=True,
    )
    pairs = match_controls(df, onsets(A=10))
    assert pairs.columns.tolist() == ["patient_id", "control_id", "onset_hour"]
    assert pairs["control_id"].tolist() == ["c4"]
    assert pairs["onset_hour"].tolist() == [10.0]


def test_each_control_is_used_at_most_once():
    df = pd.concat(
        [stay("A", 20, 4), stay("B", 20, 4), stay("c1", 21), stay("c2", 25)], ignore_index=True
    )
    pairs = match_controls(df, onsets(A=10, B=10))
    assert sorted(pairs["control_id"]) == ["c1", "c2"]


def test_septic_patients_without_an_available_control_are_left_unmatched():
    df = pd.concat([stay("A", 20, 4), stay("B", 20, 4), stay("c1", 21)], ignore_index=True)
    pairs = match_controls(df, onsets(A=10, B=10))
    assert pairs["control_id"].isna().sum() == 1
    assert pairs["control_id"].notna().sum() == 1

    nobody = match_controls(pd.concat([stay("A", 20, 4), stay("c1", 5)], ignore_index=True), onsets(A=10))
    assert nobody["control_id"].isna().all()   # the only control left before the onset


def test_other_septic_patients_are_never_used_as_controls():
    df = pd.concat(
        [stay("A", 20, 4), stay("other_septic", 20, 4), stay("c1", 40)], ignore_index=True
    )
    pairs = match_controls(df, onsets(A=10))
    assert pairs["control_id"].tolist() == ["c1"]


def test_matching_is_reproducible_and_ties_depend_on_the_seed():
    df = pd.concat([stay("A", 20, 4)] + [stay(f"c{i}", 22) for i in range(20)], ignore_index=True)
    first = match_controls(df, onsets(A=10), seed=1)["control_id"].iloc[0]
    assert match_controls(df, onsets(A=10), seed=1)["control_id"].iloc[0] == first
    others = {match_controls(df, onsets(A=10), seed=s)["control_id"].iloc[0] for s in range(2, 12)}
    assert others - {first}   # a different seed can pick a different one of the tied controls


def test_match_controls_rejects_bad_onset_tables():
    df = pd.concat([stay("A", 20, 4), stay("c1", 21)], ignore_index=True)
    with pytest.raises(ValueError, match="not in df"):
        match_controls(df, onsets(Z=10))
    with pytest.raises(ValueError, match="not septic"):
        match_controls(df, onsets(c1=10))


# ---------- score_trajectories ----------

def trajectory_data():
    a = stay("A", 40, label_from=24)
    a["score"] = a["ICULOS"] / 100                  # score at hour h is h/100
    c = stay("C", 35)
    c["score"] = 0.2
    c.loc[c["ICULOS"] == 30, "score"] = 0.9          # a spike at the pseudo-onset hour
    return pd.concat([a, c], ignore_index=True)


def one_pair(onset=30.0):
    return pd.DataFrame({"patient_id": ["A"], "control_id": ["C"], "onset_hour": [onset]})


def test_trajectory_is_aligned_on_the_onset_hour():
    t = score_trajectories(trajectory_data(), "score", one_pair())
    sep = t[t["group"] == "sepsis"].set_index("offset")
    ctl = t[t["group"] == "control"].set_index("offset")
    assert sep.index.min() == -24 and sep.index.max() == 0 and len(sep) == 25
    assert sep.loc[0, "mean"] == pytest.approx(0.30)     # hour 30
    assert sep.loc[-24, "mean"] == pytest.approx(0.06)   # hour 6
    assert ctl.loc[0, "mean"] == pytest.approx(0.9)      # control's pseudo-onset hour
    assert ctl.loc[-5, "mean"] == pytest.approx(0.2)
    assert (sep["n"] == 1).all()


def test_trajectory_is_cut_short_when_the_record_starts_late():
    df = trajectory_data()
    t = score_trajectories(df, "score", one_pair(onset=10.0))   # only 9 hours before hour 10
    sep = t[t["group"] == "sepsis"]
    assert sep["offset"].min() == -9 and len(sep) == 10


def test_trajectory_window_can_be_shortened():
    t = score_trajectories(trajectory_data(), "score", one_pair(), window_hours=3)
    assert t.loc[t["group"] == "sepsis", "offset"].tolist() == [-3, -2, -1, 0]


def test_trajectory_statistics_across_patients():
    frames = []
    for pid, value in (("A", 0.2), ("B", 0.4), ("D", 0.6), ("E", 0.8)):
        p = stay(pid, 30, label_from=24)
        p["score"] = value
        frames.append(p)
    for pid in ("c1", "c2", "c3", "c4"):
        p = stay(pid, 35)
        p["score"] = 0.1
        frames.append(p)
    df = pd.concat(frames, ignore_index=True)
    pairs = pd.DataFrame(
        {"patient_id": ["A", "B", "D", "E"], "control_id": ["c1", "c2", "c3", "c4"], "onset_hour": [30.0] * 4}
    )
    row = score_trajectories(df, "score", pairs)
    sep0 = row[(row["group"] == "sepsis") & (row["offset"] == 0)].iloc[0]
    assert sep0["n"] == 4
    assert sep0["mean"] == pytest.approx(0.5)
    assert sep0["median"] == pytest.approx(0.5)
    assert sep0["q25"] == pytest.approx(0.35)
    assert sep0["q75"] == pytest.approx(0.65)


def test_unmatched_patients_are_dropped_from_both_groups():
    df = trajectory_data()
    pairs = pd.concat([one_pair(), pd.DataFrame({"patient_id": ["B"], "control_id": [None], "onset_hour": [20.0]})])
    t = score_trajectories(df, "score", pairs)
    assert set(t["group"]) == {"sepsis", "control"}
    assert (t["n"] == 1).all()


def test_missing_scores_are_not_counted():
    df = trajectory_data()
    df.loc[(df["patient_id"] == "A") & (df["ICULOS"] == 30), "score"] = np.nan
    t = score_trajectories(df, "score", one_pair())
    sep = t[t["group"] == "sepsis"].set_index("offset")
    assert 0 not in sep.index


def test_trajectory_rejects_negative_window():
    with pytest.raises(ValueError):
        score_trajectories(trajectory_data(), "score", one_pair(), window_hours=-1)


def test_lead_time_matching_and_trajectories_work_together():
    df = cohort()
    table = patient_lead_times(df, "score", 0.5)
    table = table[~table["left_censored"]]
    pairs = match_controls(df, table)
    t = score_trajectories(df, "score", pairs)
    assert pairs["control_id"].isin(["c1", "c2"]).all() or pairs["control_id"].isna().any()
    assert set(t["group"]) <= {"sepsis", "control"}
    assert t["offset"].between(-24, 0).all()


# ---------- Challenge utility ----------

def test_constants_match_the_official_challenge_scoring_code():
    assert (DT_EARLY, DT_OPTIMAL, DT_LATE) == (-12, -6, 3)
    assert (MAX_U_TP, MIN_U_FN, U_FP, U_TN) == (1, -2, -0.05, 0)


def gain_from_one_alert(dt, n=40, onset=25.0):
    """Utility change from alerting only at hour onset+dt, compared with never alerting."""
    hours = np.arange(1, n + 1)
    alert = (hours == onset + dt).astype(int)
    return prediction_utility(hours, alert, onset) - prediction_utility(hours, np.zeros(n, int), onset)


@pytest.mark.parametrize(
    "dt, expected",
    [
        (-20, -0.05),   # far too early: costs the false-alarm penalty
        (-13, -0.05),
        (-12, 0.0),     # the earliest hour that earns credit
        (-9, 0.5),
        (-6, 1.0),      # the optimal hour
        (-2, 13 / 9),   # reward 5/9 plus avoiding a missed-hour penalty of 8/9
        (3, 2.0),       # reward 0, but avoids the full -2 miss penalty
        (4, 0.0),       # more than 3 hours after onset counts for nothing
    ],
)
def test_utility_of_a_single_alert(dt, expected):
    assert gain_from_one_alert(dt) == pytest.approx(expected)


def test_utility_of_never_alerting_and_of_perfect_alerting_for_one_septic_patient():
    hours = np.arange(1, 41)
    onset = 25.0
    assert prediction_utility(hours, np.zeros(40, int), onset) == pytest.approx(-10.0)
    perfect = ((hours >= onset - 12) & (hours <= onset + 3)).astype(int)
    assert prediction_utility(hours, perfect, onset) == pytest.approx(7.5)


def test_every_alerting_hour_costs_a_never_septic_patient():
    hours = np.arange(1, 11)
    alerts = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    assert prediction_utility(hours, alerts, None) == pytest.approx(-0.15)
    assert prediction_utility(hours, np.zeros(10, int), None) == 0.0


def test_prediction_utility_rejects_length_mismatch():
    with pytest.raises(ValueError):
        prediction_utility([1, 2, 3], [0, 1], 5.0)


def reference_utility(labels, predictions):
    """Slow row-by-row version of the piecewise utility, used to check the fast one."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    septic = bool(labels.any())
    t_sepsis = int(np.argmax(labels)) + 6 if septic else None
    total = 0.0
    for t in range(len(labels)):
        if not septic:
            total += -0.05 if predictions[t] else 0.0
            continue
        dt = t - t_sepsis
        if dt > 3:
            continue
        if predictions[t]:
            total += max((dt + 12) / 6.0, -0.05) if dt <= -6 else (3 - dt) / 9.0
        elif dt > -6:
            total += -2 * (dt + 6) / 9.0
    return total


def random_cohort(n_patients=400, seed=0):
    rng = np.random.default_rng(seed)
    frames, ref = [], {"observed": 0.0, "best": 0.0, "inaction": 0.0}
    for i in range(n_patients):
        n = int(rng.integers(5, 70))
        labels = np.zeros(n, dtype=int)
        if rng.random() < 0.4:
            labels[int(rng.integers(0, n)):] = 1
        predictions = (rng.random(n) < rng.choice([0.05, 0.3, 0.8])).astype(int)

        best = np.zeros(n, dtype=int)
        if labels.any():
            t_sepsis = int(np.argmax(labels)) + 6
            best[max(0, t_sepsis - 12): min(t_sepsis + 3 + 1, n)] = 1
        ref["observed"] += reference_utility(labels, predictions)
        ref["best"] += reference_utility(labels, best)
        ref["inaction"] += reference_utility(labels, np.zeros(n, dtype=int))

        frames.append(
            pd.DataFrame(
                {"patient_id": f"p{i}", "ICULOS": np.arange(1, n + 1), "SepsisLabel": labels,
                 "score": predictions.astype(float)}
            )
        )
    return pd.concat(frames, ignore_index=True), ref


def test_fast_utility_matches_the_slow_reference_on_random_patients():
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(5, 70))
        labels = np.zeros(n, dtype=int)
        if rng.random() < 0.5:
            labels[int(rng.integers(0, n)):] = 1
        predictions = (rng.random(n) < 0.3).astype(int)
        hours = np.arange(1, n + 1)
        onset = float(hours[np.argmax(labels)] + 6) if labels.any() else None
        assert prediction_utility(hours, predictions, onset) == pytest.approx(
            reference_utility(labels, predictions)
        )


def test_cohort_utility_matches_the_reference():
    df, ref = random_cohort()
    out = normalized_utility(df, "score", threshold=0.5)
    assert out["observed_utility"] == pytest.approx(ref["observed"])
    assert out["best_utility"] == pytest.approx(ref["best"])
    assert out["inaction_utility"] == pytest.approx(ref["inaction"])
    expected = (ref["observed"] - ref["inaction"]) / (ref["best"] - ref["inaction"])
    assert out["normalized_utility"] == pytest.approx(expected)


def test_normalized_utility_is_one_for_perfect_alerts_and_zero_for_none():
    df, _ = random_cohort(100, seed=2)
    onset = df.loc[df["SepsisLabel"] == 1].groupby("patient_id")["ICULOS"].min() + 6
    dt = df["ICULOS"] - df["patient_id"].map(onset)
    perfect = df.assign(score=((dt >= -12) & (dt <= 3)).astype(float))
    assert normalized_utility(perfect, "score", 0.5)["normalized_utility"] == pytest.approx(1.0)
    assert normalized_utility(df, "score", threshold=2.0)["normalized_utility"] == pytest.approx(0.0)


def test_alerting_only_at_the_wrong_times_scores_below_zero():
    df, _ = random_cohort(100, seed=3)
    onset = df.loc[df["SepsisLabel"] == 1].groupby("patient_id")["ICULOS"].min() + 6
    dt = df["ICULOS"] - df["patient_id"].map(onset)
    worst = df.assign(score=1.0 - ((dt >= -12) & (dt <= 3)).astype(float))
    assert normalized_utility(worst, "score", 0.5)["normalized_utility"] < 0


def test_normalized_utility_with_no_septic_patients_is_nan():
    out = normalized_utility(patient("c1"), "score", 0.5)
    assert np.isnan(out["normalized_utility"])


def test_normalized_utility_rejects_bad_input():
    with pytest.raises(ValueError, match="score"):
        normalized_utility(cohort().drop(columns=["score"]), "score", 0.5)
    bad = cohort()
    bad.loc[0, "SepsisLabel"] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        normalized_utility(bad, "score", 0.5)
