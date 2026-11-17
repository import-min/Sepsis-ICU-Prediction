"""Error-mode analysis (study_design.md, section 9).

All functions work on the row-level DataFrame used elsewhere in the repo (one
row per patient-hour: patient_id, ICULOS, SepsisLabel, a score column, the
static columns Age, Gender, Unit1, Unit2, and the vitals and labs).

Contents:
  patient_table / add_subgroups   one row per patient; age, sex, unit, stay length,
                                  and amount-of-missing-data groups
  subgroup_metrics                sensitivity, PPV, calibration for each subgroup
  false_alarm_table               alerts per patient-day by ICU unit (never-septic patients)
  false_alarm_profile             what high-scoring never-septic patients look like
  miss_profile                    what missed septic patients look like vs detected ones
  label_noise_candidates          cases where the label itself looks questionable
  build_review_table              sample of error cases for manual review
  update_review_file              write the review table without overwriting your notes
  shift_comparison                performance and calibration on hospital A vs B

Things to verify or log in the decisions log (study_design.md):
  * Gender is read as 0 = female, 1 = male; Unit1 = 1 as "MICU" and Unit2 = 1 as
    "SICU". Check these against the Challenge data description and edit the
    mapping in patient_table if they differ.
  * The data has no field for infection source or post-operative status, so
    "what do the high-scoring patients look like" can only be answered from the
    vitals, labs, and how often they were measured.
  * "Missing data amount" is the fraction of measurement cells that are empty
    over the whole stay, split into tertiles within the data you pass in.
  * Subgroup tables are hour-level (plus optional patient-level sensitivity) and
    have no bootstrap intervals. Groups with few septic patients are flagged
    (low_n) rather than hidden.
  * A sustained run of alerts in a never-septic patient is only a CANDIDATE for
    a missed label. Someone has to look at the chart.
  * Runs of alerts assume consecutive hourly rows, as elsewhere in the repo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sepsis_icu.metrics import METRIC_NAMES, evaluate

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_TIME_COLUMN = "ICULOS"
DEFAULT_LABEL_COLUMN = "SepsisLabel"

AGE_COLUMN = "Age"
SEX_COLUMN = "Gender"
UNIT1_COLUMN = "Unit1"
UNIT2_COLUMN = "Unit2"

REVIEW_COLUMNS = [
    "patient_id", "error_type", "reason", "max_score", "alert_hours", "stay_hours",
    "reviewer", "date_reviewed", "label_questionable", "finding", "notes",
]


def _require(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


# ---------- patients and subgroups ----------

def patient_table(
    df: pd.DataFrame,
    measurement_columns: Sequence[str],
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """One row per patient.

    Columns: id_col, age, sex, unit, stay_hours (rows), missing_fraction (share
    of empty cells in measurement_columns over the stay), ever_septic.
    Missing or absent static columns give NaN age and "unknown" sex or unit.
    """
    measurement_columns = list(measurement_columns)
    if not measurement_columns:
        raise ValueError("measurement_columns is empty")
    _require(df, [id_col, time_col, label_col, *measurement_columns])

    grouped = df.groupby(id_col, sort=True)
    stay = grouped.size()

    def static(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(grouped[name].first(), errors="coerce")
        return pd.Series(np.nan, index=stay.index)

    age, gender, unit1, unit2 = static(AGE_COLUMN), static(SEX_COLUMN), static(UNIT1_COLUMN), static(UNIT2_COLUMN)
    sex = gender.map({0.0: "female", 1.0: "male"}).fillna("unknown")
    unit = pd.Series(
        np.select([unit1 == 1, unit2 == 1], ["MICU", "SICU"], default="unknown"), index=stay.index
    )
    missing_fraction = (
        df[measurement_columns].isna().groupby(df[id_col]).mean().mean(axis=1).reindex(stay.index)
    )
    ever_septic = grouped[label_col].max() == 1

    out = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "unit": unit,
            "stay_hours": stay,
            "missing_fraction": missing_fraction,
            "ever_septic": ever_septic,
        }
    )
    out.index.name = id_col
    return out.reset_index()


def _edge_labels(edges: Sequence[float], suffix: str = "") -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0 and lo == 0:
            labels.append(f"<{hi:g}{suffix}")
        elif math.isinf(hi):
            labels.append(f"{lo:g}{suffix}+")
        else:
            labels.append(f"{lo:g}-{hi - 1:g}{suffix}")
    return labels


def _cut(values: pd.Series, edges: Sequence[float], suffix: str = "") -> pd.Series:
    cut = pd.cut(values, bins=list(edges), right=False, labels=_edge_labels(edges, suffix))
    return cut.astype(object).where(cut.notna(), "unknown")


def add_subgroups(
    patients: pd.DataFrame,
    age_edges: Sequence[float] = (0, 40, 65, 80, np.inf),
    stay_edges: Sequence[float] = (0, 24, 72, 168, np.inf),
) -> pd.DataFrame:
    """Add age_group, stay_group and missing_group columns (returns a copy).

    Age groups default to <40, 40-64, 65-79, 80+ and stay groups to <24h,
    24-71h, 72-167h, 168h+. missing_group is low / medium / high: tertiles of
    missing_fraction within the patients given. Unknown values are "unknown".
    """
    _require(patients, ["age", "stay_hours", "missing_fraction"])
    out = patients.copy()
    out["age_group"] = _cut(out["age"], age_edges)
    out["stay_group"] = _cut(out["stay_hours"], stay_edges, "h")

    rank = out["missing_fraction"].rank(method="first")
    if rank.notna().sum() >= 3:
        tertile = pd.qcut(rank, 3, labels=["low", "medium", "high"])
        out["missing_group"] = tertile.astype(object).where(tertile.notna(), "unknown")
    else:
        out["missing_group"] = "unknown"
    return out


def subgroup_metrics(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    group_col: str,
    score_col: str,
    threshold: float,
    is_probability: bool = True,
    lead_table: Optional[pd.DataFrame] = None,
    min_septic_patients: int = 20,
    id_col: str = DEFAULT_ID_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Metrics at a fixed threshold for each value of a patient-level group column.

    Columns: variable, group, n_patients, n_septic_patients, n_hours, then every
    metric in sepsis_icu.metrics.METRIC_NAMES (prevalence, auroc, auprc, brier,
    calibration_intercept, calibration_slope, ece, sensitivity, ppv,
    alerts_per_patient_day), then patient_sensitivity if lead_table is given
    (share of septic patients first alerted before or at onset; left-censored
    patients excluded), and low_n (True if fewer than min_septic_patients septic
    patients). Patients missing from `patients` fall in group "unknown".
    """
    _require(df, [id_col, label_col, score_col])
    _require(patients, [id_col, group_col])

    lookup = patients.drop_duplicates(id_col).set_index(id_col)[group_col]
    row_group = df[id_col].map(lookup)
    row_group = row_group.astype(object).where(row_group.notna(), "unknown")

    lead_group = None
    if lead_table is not None:
        _require(lead_table, [id_col, "lead_time_hours", "left_censored"])
        lt = lead_table[~lead_table["left_censored"].astype(bool)]
        mapped = lt[id_col].map(lookup)
        lead_group = pd.DataFrame(
            {"group": mapped.astype(object).where(mapped.notna(), "unknown"), "on_time": lt["lead_time_hours"] >= 0}
        )

    rows = []
    for group in sorted(row_group.unique(), key=str):
        mask = (row_group == group).to_numpy()
        y = df.loc[mask, label_col].to_numpy()
        s = df.loc[mask, score_col].to_numpy(dtype=float)
        n_septic = int(df.loc[mask & (df[label_col] == 1).to_numpy(), id_col].nunique())
        row = {
            "variable": group_col,
            "group": group,
            "n_patients": int(df.loc[mask, id_col].nunique()),
            "n_septic_patients": n_septic,
            "n_hours": int(mask.sum()),
            **evaluate(y, s, threshold, is_probability),
        }
        if lead_group is not None:
            members = lead_group[lead_group["group"] == group]["on_time"]
            row["patient_sensitivity"] = float(members.mean()) if len(members) else float("nan")
        row["low_n"] = n_septic < min_septic_patients
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- false alarms ----------

def false_alarm_table(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    score_col: str,
    threshold: float,
    by: str = "unit",
    id_col: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Alerts in patients who never became septic, by a patient-level column.

    Columns: group, n_patients, n_hours, alert_hours, alerts_per_patient_day,
    pct_patients_with_alert. The last row, "all", covers every group.
    """
    _require(df, [id_col, score_col])
    _require(patients, [id_col, by, "ever_septic"])

    never = patients.loc[~patients["ever_septic"].astype(bool), [id_col, by]]
    rows = df[df[id_col].isin(never[id_col])].copy()
    rows["_group"] = rows[id_col].map(never.set_index(id_col)[by])
    rows["_alert"] = rows[score_col].to_numpy(dtype=float) >= threshold

    def summarize(label, part):
        n_hours = len(part)
        alert_hours = int(part["_alert"].sum())
        per_patient = part.groupby(id_col)["_alert"].any()
        return {
            "group": label,
            "n_patients": int(part[id_col].nunique()),
            "n_hours": n_hours,
            "alert_hours": alert_hours,
            "alerts_per_patient_day": alert_hours / (n_hours / 24) if n_hours else float("nan"),
            "pct_patients_with_alert": float(100 * per_patient.mean()) if len(per_patient) else float("nan"),
        }

    out = [summarize(g, part) for g, part in rows.groupby("_group", sort=True)]
    out.append(summarize("all", rows))
    return pd.DataFrame(out)


# ---------- profiles: what error cases look like ----------

def patient_profiles(
    df: pd.DataFrame,
    columns: Sequence[str],
    up_to_hour: Optional[Mapping] = None,
    observed_fraction: bool = False,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    """Per-patient median of each column (indexed by patient).

    up_to_hour: mapping patient -> last hour to include (for example onset
        hours); patients not in it keep all their rows.
    observed_fraction: also add "<col>_observed_frac", the share of hours in
        which the column was measured (a proxy for how closely the patient
        was being worked up).
    """
    columns = list(columns)
    _require(df, [id_col, time_col, *columns])
    data = df
    if up_to_hour is not None:
        cutoff = df[id_col].map(up_to_hour)
        data = df[cutoff.isna() | (df[time_col] <= cutoff)]
    profiles = data.groupby(id_col)[columns].median()
    if observed_fraction:
        frac = data[columns].notna().groupby(data[id_col]).mean().add_suffix("_observed_frac")
        profiles = profiles.join(frac)
    return profiles


def compare_profiles(
    profiles: pd.DataFrame,
    ids_a: Sequence,
    ids_b: Sequence,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Compare two groups of patients feature by feature.

    Columns: feature, n_<a>, n_<b>, median_<a>, median_<b>, standardized_difference
    ((mean a - mean b) / pooled standard deviation; about 0.2 is small, 0.5
    medium, 0.8 large). Sorted by the size of the difference, largest first.
    """
    a = profiles.loc[profiles.index.intersection(list(ids_a))]
    b = profiles.loc[profiles.index.intersection(list(ids_b))]

    rows = []
    for feature in profiles.columns:
        xa, xb = a[feature].dropna(), b[feature].dropna()
        pooled = math.sqrt((xa.var() + xb.var()) / 2) if len(xa) > 1 and len(xb) > 1 else float("nan")
        smd = (xa.mean() - xb.mean()) / pooled if pooled and not math.isnan(pooled) else float("nan")
        rows.append(
            {
                "feature": feature,
                f"n_{label_a}": int(len(xa)),
                f"n_{label_b}": int(len(xb)),
                f"median_{label_a}": float(xa.median()) if len(xa) else float("nan"),
                f"median_{label_b}": float(xb.median()) if len(xb) else float("nan"),
                "standardized_difference": float(smd),
            }
        )
    out = pd.DataFrame(rows)
    order = out["standardized_difference"].abs().sort_values(ascending=False, na_position="last").index
    return out.loc[order].reset_index(drop=True)


def false_alarm_profile(
    df: pd.DataFrame,
    patients: pd.DataFrame,
    score_col: str,
    columns: Sequence[str],
    top_fraction: float = 0.05,
    observed_fraction: bool = False,
    id_col: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Compare the highest-scoring never-septic patients with the other never-septic patients.

    "High-scoring" = the top_fraction of never-septic patients by their peak score
    (at least one patient). Output is from compare_profiles, with groups
    labelled high_scoring and other_non_septic.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be between 0 (exclusive) and 1")
    _require(df, [id_col, score_col])
    _require(patients, [id_col, "ever_septic"])

    never_ids = patients.loc[~patients["ever_septic"].astype(bool), id_col]
    rows = df[df[id_col].isin(never_ids)]
    peak = rows.groupby(id_col)[score_col].max().dropna()
    n_high = max(1, int(math.ceil(top_fraction * len(peak))))
    ordered = peak.sort_values(ascending=False, kind="stable").index
    profiles = patient_profiles(rows, columns, observed_fraction=observed_fraction, id_col=id_col)
    return compare_profiles(profiles, ordered[:n_high], ordered[n_high:], "high_scoring", "other_non_septic")


def miss_profile(
    df: pd.DataFrame,
    lead_table: pd.DataFrame,
    columns: Sequence[str],
    observed_fraction: bool = False,
    id_col: str = DEFAULT_ID_COLUMN,
) -> pd.DataFrame:
    """Compare missed septic patients (never alerted, or alerted only after onset)
    with detected ones (first alert at or before onset).

    Uses each patient's rows up to the onset hour. Left-censored patients are
    left out. Output is from compare_profiles, with groups labelled missed
    and detected.
    """
    _require(lead_table, [id_col, "onset_hour", "lead_time_hours", "left_censored"])
    t = lead_table[~lead_table["left_censored"].astype(bool)]
    on_time = t["lead_time_hours"] >= 0  # NaN (never alerted) counts as not on time
    detected, missed = t.loc[on_time, id_col], t.loc[~on_time, id_col]

    septic_rows = df[df[id_col].isin(t[id_col])]
    profiles = patient_profiles(
        septic_rows, columns, up_to_hour=t.set_index(id_col)["onset_hour"],
        observed_fraction=observed_fraction, id_col=id_col,
    )
    return compare_profiles(profiles, missed, detected, "missed", "detected")


# ---------- label noise ----------

def label_noise_candidates(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    lead_table: pd.DataFrame,
    min_alert_run: int = 12,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Cases where the label itself looks questionable. Columns: id_col, flag, detail.

    left_censored_onset         label already 1 at the first row (detail: onset hour)
    label_turns_off             label goes back to 0 after turning on (detail: first label hour)
    onset_after_record_end      reconstructed onset is after the patient's last row
                                (detail: hours after the record ends)
    sustained_alerts_never_septic  never-septic patient with min_alert_run or more
                                consecutive alert hours (detail: longest run); a
                                candidate for sepsis that was never labelled
    """
    _require(df, [id_col, time_col, label_col, score_col])
    _require(lead_table, [id_col, "onset_hour", "first_label_hour", "left_censored", "label_monotone"])

    last_hour = df.groupby(id_col)[time_col].max()
    rows = []
    for r in lead_table.itertuples(index=False):
        pid = getattr(r, id_col)
        if r.left_censored:
            rows.append((pid, "left_censored_onset", float(r.onset_hour)))
        if not r.label_monotone:
            rows.append((pid, "label_turns_off", float(r.first_label_hour)))
        if r.onset_hour > last_hour.loc[pid]:
            rows.append((pid, "onset_after_record_end", float(r.onset_hour - last_hour.loc[pid])))

    d = df.sort_values([id_col, time_col], kind="stable")
    ids = d[id_col].to_numpy()
    alert = d[score_col].to_numpy(dtype=float) >= threshold
    never = (d.groupby(id_col)[label_col].transform("max") != 1).to_numpy()
    if len(d):
        new_run = np.ones(len(d), dtype=bool)
        new_run[1:] = (alert[1:] != alert[:-1]) | (ids[1:] != ids[:-1])
        runs = pd.DataFrame({"run": np.cumsum(new_run), "pid": ids, "alert": alert, "never": never})
        runs = runs.groupby("run").agg(pid=("pid", "first"), alert=("alert", "first"), never=("never", "first"), length=("run", "size"))
        longest = runs[runs["alert"] & runs["never"]].groupby("pid")["length"].max()
        for pid, length in longest[longest >= min_alert_run].items():
            rows.append((pid, "sustained_alerts_never_septic", float(length)))

    out = pd.DataFrame(rows, columns=[id_col, "flag", "detail"])
    return out.sort_values(["flag", id_col], kind="stable").reset_index(drop=True)


# ---------- manual review ----------

def build_review_table(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    lead_table: pd.DataFrame,
    per_type: int = 10,
    seed: int = 42,
    min_alert_run: int = 12,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """A sample of error cases to review by hand, with empty columns to fill in.

    error_type:
      false_alarm          the per_type never-septic patients with the highest peak score
      never_alerted        random septic patients who were never alerted
      alerted_after_onset  random septic patients first alerted after onset
      label_noise          up to per_type patients per flag from label_noise_candidates
    Left-censored patients are left out of the two septic samples.
    Columns: REVIEW_COLUMNS (reviewer, date_reviewed, label_questionable,
    finding and notes start empty).
    """
    if per_type < 1:
        raise ValueError("per_type must be at least 1")
    _require(df, [id_col, time_col, label_col, score_col])
    _require(lead_table, [id_col, "lead_time_hours", "left_censored"])

    grouped = df.groupby(id_col)
    is_alert = pd.Series(df[score_col].to_numpy(dtype=float) >= threshold, index=df.index)
    summary = pd.DataFrame(
        {
            "max_score": grouped[score_col].max(),
            "alert_hours": is_alert.groupby(df[id_col]).sum().astype(int),
            "stay_hours": grouped.size(),
        }
    )
    ever_septic = grouped[label_col].max() == 1
    rng = np.random.default_rng(seed)

    picked = []  # (patient, error_type, reason)
    never_septic = summary.loc[~ever_septic.reindex(summary.index)].sort_values("max_score", ascending=False, kind="stable")
    for pid in never_septic.index[:per_type]:
        picked.append((pid, "false_alarm", "highest-scoring patient who never became septic"))

    t = lead_table[~lead_table["left_censored"].astype(bool)]
    groups = (
        ("never_alerted", t.loc[t["lead_time_hours"].isna(), id_col], "septic patient who was never alerted"),
        ("alerted_after_onset", t.loc[t["lead_time_hours"] < 0, id_col], "septic patient first alerted after onset"),
    )
    for error_type, ids, reason in groups:
        ids = np.array(sorted(ids))
        for pid in ids[rng.permutation(len(ids))[:per_type]]:
            picked.append((pid, error_type, reason))

    candidates = label_noise_candidates(df, score_col, threshold, lead_table, min_alert_run, id_col, time_col, label_col)
    for flag, part in candidates.groupby("flag", sort=True):
        chosen = part.iloc[rng.permutation(len(part))[:per_type]].sort_values(id_col)
        for r in chosen.itertuples(index=False):
            picked.append((getattr(r, id_col), "label_noise", f"{flag} ({r.detail:g})"))

    out = pd.DataFrame(picked, columns=["patient_id", "error_type", "reason"])
    out = out.join(summary[["max_score", "alert_hours", "stay_hours"]], on="patient_id")
    for col in ("reviewer", "date_reviewed", "label_questionable", "finding", "notes"):
        out[col] = ""
    return out[REVIEW_COLUMNS].reset_index(drop=True)


def update_review_file(path: str, new_rows: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """Write the review table to CSV WITHOUT overwriting existing review notes.

    If the file exists, its rows are kept exactly as they are and only rows with
    a (patient, error_type) pair not already in the file are added. Returns the
    table that was written.
    """
    _require(new_rows, [id_col, "error_type"])
    path = Path(path)
    fresh = new_rows.astype(object).where(new_rows.notna(), "")

    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        _require(existing, [id_col, "error_type"])
        have = set(zip(existing[id_col], existing["error_type"]))
        keep = [(p, t) not in have for p, t in zip(fresh[id_col].astype(str), fresh["error_type"].astype(str))]
        merged = pd.concat([existing, fresh[keep].astype(str)], ignore_index=True).fillna("")
    else:
        merged = fresh

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


# ---------- shift between hospitals ----------

def shift_comparison(
    y_a: Sequence,
    scores_a: Sequence,
    y_b: Sequence,
    scores_b: Sequence,
    threshold: float,
    is_probability: bool = True,
) -> pd.DataFrame:
    """Every metric on hospital A and hospital B at the same fixed threshold.

    Columns: metric, hospital_a, hospital_b, difference (B minus A). Use the
    threshold chosen on development data. alerts_per_patient_day shows how much
    the alert rate moved; the calibration rows show calibration shift. Put
    bootstrap intervals on each side with sepsis_icu.metrics.bootstrap_ci.
    """
    a = evaluate(y_a, scores_a, threshold, is_probability)
    b = evaluate(y_b, scores_b, threshold, is_probability)
    return pd.DataFrame(
        {
            "metric": list(METRIC_NAMES),
            "hospital_a": [a[m] for m in METRIC_NAMES],
            "hospital_b": [b[m] for m in METRIC_NAMES],
            "difference": [b[m] - a[m] for m in METRIC_NAMES],
        }
    )
