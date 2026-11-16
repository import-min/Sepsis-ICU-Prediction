"""Early-vs-late analysis across patients (study_design.md, section 6).

Builds on sepsis_icu.onset, which handles one patient at a time. All functions
work on a row-level DataFrame with one row per patient-hour:

    patient_id, ICULOS (hour), SepsisLabel, and a score column

Contents:
  patient_lead_times   one row per septic patient: onset, first alert, lead time, bin
  bin_summary          patients per lead-time bin with 95% bootstrap CIs
  detection_summary    patient-level sensitivity, alerted-after-onset, never alerted
  match_controls       match each septic patient to a non-septic control
  score_trajectories   mean/median score in the 24 hours before onset, sepsis vs controls
  normalized_utility   the PhysioNet/CinC 2019 Challenge utility score

Definitions to log in the decisions log (study_design.md):
  * First alert = the first hour with score >= threshold, anywhere in the stay.
    The threshold is fixed beforehand on development data (see
    sepsis_icu.metrics.threshold_for_alert_rate) and carried unchanged.
  * Lead time = onset hour - first alert hour. Alerts after onset are "missed".
  * Patients whose label is already 1 at their first row have a left-censored
    onset (the true onset may be earlier). They are excluded from the bin
    summary by default; patient_lead_times keeps them with a flag.
  * Controls: for each septic patient, the unused non-septic patient who is
    still in the ICU at the septic patient's onset hour and whose total stay
    length is closest. A control's "pseudo-onset" is the matched patient's
    onset hour, so both are viewed from the same point in the ICU stay.
    Each control is used at most once; ties are broken at random with a fixed
    seed. Septic patients with no eligible control stay unmatched.
  * The trajectory tables are descriptive (mean, median, quartiles). They do
    not include bootstrap intervals.

Utility score: the constants and piecewise function follow the official scoring
code of the 2019 Challenge (evaluate_sepsis_score.py) and were checked against
it. The code treats consecutive hourly rows as the official code does, so
"hour - onset" matches its "row - t_sepsis". A positive prediction is
score >= threshold; predictions are binary. Alerts more than 12 hours before
onset cost 0.05 each, alerts from 12 to 6 hours before earn partial credit up
to 1, alerts after the optimal time earn less, missed hours between 6 hours
before and 3 hours after onset are penalized up to -2, and hours more than 3
hours after onset count for nothing. For patients who never develop sepsis,
each alerting hour costs 0.05. The normalized score is
(observed - no alerts) / (perfect - no alerts): 1 is perfect, 0 is no better
than never alerting, and it can be negative.

Caution: false alarms are penalized very lightly (0.05 per hour), so a model
that alerts constantly can still score well. That is why this is a secondary
metric. Always report it next to alerts per patient-day
(sepsis_icu.metrics.operating_point).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sepsis_icu.onset import (
    BIN_EARLY,
    BIN_INTERMEDIATE,
    BIN_LATE,
    BIN_MISSED,
    LABEL_SHIFT_HOURS,
    assign_lead_time_bin,
    first_alert_hour,
    reconstruct_onset,
)

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_TIME_COLUMN = "ICULOS"
DEFAULT_LABEL_COLUMN = "SepsisLabel"

BIN_ORDER = (BIN_EARLY, BIN_INTERMEDIATE, BIN_LATE, BIN_MISSED)

# Challenge utility constants (hours relative to onset, and utility values).
DT_EARLY = -12.0
DT_OPTIMAL = -6.0
DT_LATE = 3.0
MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0


def _require(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


# ---------- lead time per patient ----------

def patient_lead_times(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    shift: float = LABEL_SHIFT_HOURS,
    early_min: float = 12.0,
    intermediate_min: float = 6.0,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """One row per septic patient.

    Columns: id_col, onset_hour, first_label_hour, left_censored, label_monotone,
    first_alert_hour (NaN if never alerted), lead_time_hours (NaN if never
    alerted; negative if the first alert came after onset), bin.
    Patients who never become septic are not in the table.
    """
    _require(df, [id_col, time_col, label_col, score_col])
    df = df.sort_values([id_col, time_col], kind="stable")
    septic_ids = df.loc[df[label_col] == 1, id_col].unique()
    septic = df[df[id_col].isin(septic_ids)]

    rows = []
    for pid, g in septic.groupby(id_col, sort=True):
        hours = g[time_col].to_numpy(dtype=float)
        onset = reconstruct_onset(hours, g[label_col].to_numpy(), shift)
        alert = first_alert_hour(hours, g[score_col].to_numpy(dtype=float), threshold)
        rows.append(
            {
                id_col: pid,
                "onset_hour": onset.onset_hour,
                "first_label_hour": onset.first_label_hour,
                "left_censored": onset.left_censored,
                "label_monotone": onset.label_monotone,
                "first_alert_hour": np.nan if alert is None else alert,
                "lead_time_hours": np.nan if alert is None else onset.onset_hour - alert,
                "bin": assign_lead_time_bin(onset.onset_hour, alert, early_min, intermediate_min),
            }
        )
    columns = [
        id_col, "onset_hour", "first_label_hour", "left_censored", "label_monotone",
        "first_alert_hour", "lead_time_hours", "bin",
    ]
    out = pd.DataFrame(rows, columns=columns)
    out["left_censored"] = out["left_censored"].astype(bool)
    out["label_monotone"] = out["label_monotone"].astype(bool)
    return out


def _included(table: pd.DataFrame, exclude_left_censored: bool) -> pd.DataFrame:
    _require(table, ["left_censored", "bin"])
    if not exclude_left_censored:
        return table
    return table[~table["left_censored"].astype(bool)]


def bin_summary(
    table: pd.DataFrame,
    exclude_left_censored: bool = True,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Septic patients per lead-time bin, with percentile bootstrap CIs.

    Columns: bin, n_patients, fraction, ci_lower, ci_upper. The four bins are
    always present, in the order early, intermediate, late, missed_or_post_onset.
    The CI resamples septic patients; n_resamples=0 skips it (NaN).
    """
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1 (exclusive)")
    if n_resamples < 0:
        raise ValueError("n_resamples must be zero or more")

    t = _included(table, exclude_left_censored)
    n = len(t)
    codes = t["bin"].map({b: i for i, b in enumerate(BIN_ORDER)}).to_numpy()
    if n and np.isnan(codes.astype(float)).any():
        raise ValueError("table contains an unknown bin name")
    counts = np.array([int((codes == i).sum()) for i in range(len(BIN_ORDER))])
    fraction = counts / n if n else np.full(len(BIN_ORDER), np.nan)

    lower = np.full(len(BIN_ORDER), np.nan)
    upper = np.full(len(BIN_ORDER), np.nan)
    if n and n_resamples:
        rng = np.random.default_rng(seed)
        sampled = codes[rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)]
        draws = np.stack([(sampled == i).mean(axis=1) for i in range(len(BIN_ORDER))], axis=1)
        alpha = (1 - ci_level) / 2 * 100
        lower, upper = np.percentile(draws, [alpha, 100 - alpha], axis=0)

    return pd.DataFrame(
        {
            "bin": list(BIN_ORDER),
            "n_patients": counts,
            "fraction": fraction,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    )


def detection_summary(table: pd.DataFrame, exclude_left_censored: bool = True) -> Dict[str, float]:
    """Patient-level detection numbers for septic patients.

    n_alerted_before_or_at_onset counts first alerts with lead time >= 0.
    n_alerted_only_after_onset and n_never_alerted split the missed patients.
    median_lead_time_hours is over patients alerted before or at onset.
    """
    t = _included(table, exclude_left_censored)
    n = len(t)
    lead = t["lead_time_hours"]
    on_time = lead >= 0
    return {
        "n_septic_patients": n,
        "n_left_censored_excluded": int(len(table) - n),
        "n_alerted_before_or_at_onset": int(on_time.sum()),
        "patient_sensitivity": float(on_time.sum() / n) if n else float("nan"),
        "n_alerted_only_after_onset": int((lead < 0).sum()),
        "n_never_alerted": int(lead.isna().sum()),
        "median_lead_time_hours": float(lead[on_time].median()) if on_time.any() else float("nan"),
    }


# ---------- matched controls and trajectories ----------

def match_controls(
    df: pd.DataFrame,
    onset_table: pd.DataFrame,
    seed: int = 42,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Match each septic patient to one non-septic control (see module notes).

    onset_table: one row per septic patient to match, with id_col and
        onset_hour (for example the output of patient_lead_times, optionally
        without the left-censored patients).
    Returns: id_col, control_id, onset_hour, in the order of onset_table.
    control_id is None where no eligible control was left.
    """
    _require(df, [id_col, time_col, label_col])
    _require(onset_table, [id_col, "onset_hour"])

    last_hour = df.groupby(id_col)[time_col].max()
    ever_septic = df.groupby(id_col)[label_col].max() == 1
    control_ids = last_hour.index[~ever_septic.reindex(last_hour.index).to_numpy()].to_numpy()
    control_last = last_hour.loc[control_ids].to_numpy(dtype=float)

    septic = onset_table[[id_col, "onset_hour"]].reset_index(drop=True)
    unknown = set(septic[id_col]) - set(last_hour.index)
    if unknown:
        raise ValueError(f"onset_table has patients that are not in df, for example {sorted(unknown)[:3]}")
    if not ever_septic.reindex(septic[id_col]).all():
        raise ValueError("onset_table contains patients who are not septic in df")
    septic_last = last_hour.loc[septic[id_col]].to_numpy(dtype=float)
    onset = septic["onset_hour"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(septic))
    priority = rng.permutation(len(control_ids))  # random tie-break
    available = np.ones(len(control_ids), dtype=bool)
    matched = np.full(len(septic), -1)

    for i in order:
        eligible = available & (control_last >= onset[i])
        if not eligible.any():
            continue
        gap = np.where(eligible, np.abs(control_last - septic_last[i]), np.inf)
        best = np.flatnonzero(gap == gap.min())
        j = best[np.argmin(priority[best])]
        matched[i] = j
        available[j] = False

    return pd.DataFrame(
        {
            id_col: septic[id_col],
            "control_id": [control_ids[j] if j >= 0 else None for j in matched],
            "onset_hour": septic["onset_hour"],
        }
    )


def score_trajectories(
    df: pd.DataFrame,
    score_col: str,
    pairs: pd.DataFrame,
    window_hours: float = 24.0,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    """Score by hours relative to onset, septic patients vs matched controls.

    pairs: output of match_controls. Offset 0 is the onset hour (the pseudo-onset
    for controls); negative offsets are before it.
    Columns: group ("sepsis" or "control"), offset, n, mean, median, q25, q75.
    """
    if window_hours < 0:
        raise ValueError("window_hours must be non-negative")
    _require(df, [id_col, time_col, score_col])
    _require(pairs, [id_col, "control_id", "onset_hour"])

    pairs = pairs.dropna(subset=["control_id"])
    scores = df[[id_col, time_col, score_col]].dropna(subset=[score_col])

    tables = []
    for group, key in (("sepsis", id_col), ("control", "control_id")):
        anchors = pairs[[key, "onset_hour"]].rename(columns={key: id_col})
        merged = scores.merge(anchors, on=id_col, how="inner")
        merged["offset"] = merged[time_col] - merged["onset_hour"]
        merged = merged[(merged["offset"] >= -window_hours) & (merged["offset"] <= 0)]
        g = merged.groupby("offset")[score_col]
        table = pd.DataFrame(
            {
                "n": g.count(),
                "mean": g.mean(),
                "median": g.median(),
                "q25": g.quantile(0.25),
                "q75": g.quantile(0.75),
            }
        ).reset_index()
        table.insert(0, "group", group)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ---------- Challenge utility ----------

def _row_utilities(dt: np.ndarray, predicted: np.ndarray, septic: np.ndarray) -> np.ndarray:
    """Utility of each row. dt = hour - onset (NaN for never-septic patients)."""
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL

    u = np.zeros(dt.shape, dtype=float)
    u = np.where(~septic & predicted, U_FP, u)

    in_window = septic & (dt <= DT_LATE)
    before_optimal = in_window & (dt <= DT_OPTIMAL)
    after_optimal = in_window & (dt > DT_OPTIMAL)

    u = np.where(before_optimal & predicted, np.maximum(m1 * dt + b1, U_FP), u)
    u = np.where(after_optimal & predicted, m2 * dt + b2, u)
    u = np.where(after_optimal & ~predicted, m3 * dt + b3, u)
    return u  # missed hours before the optimal time, and everything after the window, are 0


def prediction_utility(
    hours: Sequence[float],
    predictions: Sequence[int],
    onset_hour: Optional[float],
) -> float:
    """Total utility for one patient. onset_hour is None for a never-septic patient."""
    hours = np.asarray(hours, dtype=float)
    predicted = np.asarray(predictions).astype(bool)
    if hours.shape != predicted.shape:
        raise ValueError("hours and predictions must have the same length")
    septic = np.full(hours.shape, onset_hour is not None)
    dt = hours - (np.nan if onset_hour is None else onset_hour)
    return float(_row_utilities(dt, predicted, septic).sum())


def normalized_utility(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    shift: float = LABEL_SHIFT_HOURS,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> Dict[str, float]:
    """Challenge utility over the whole cohort at a fixed threshold.

    Returns observed_utility, best_utility, inaction_utility and
    normalized_utility = (observed - inaction) / (best - inaction), which is NaN
    if the data has no septic patients.
    """
    _require(df, [id_col, time_col, label_col, score_col])
    if not df[label_col].isin([0, 1]).all():
        raise ValueError(f"{label_col} must be 0 or 1 in every row")

    first_label = df.loc[df[label_col] == 1].groupby(id_col)[time_col].min()
    onset = df[id_col].map(first_label).to_numpy(dtype=float) + shift  # NaN if never septic
    septic = ~np.isnan(onset)
    dt = df[time_col].to_numpy(dtype=float) - onset
    predicted = df[score_col].to_numpy(dtype=float) >= threshold

    observed = float(_row_utilities(dt, predicted, septic).sum())
    perfect = septic & (dt >= DT_EARLY) & (dt <= DT_LATE)
    best = float(_row_utilities(dt, perfect, septic).sum())
    inaction = float(_row_utilities(dt, np.zeros(dt.shape, dtype=bool), septic).sum())

    normalized = (observed - inaction) / (best - inaction) if best != inaction else float("nan")
    return {
        "observed_utility": observed,
        "best_utility": best,
        "inaction_utility": inaction,
        "normalized_utility": normalized,
    }
