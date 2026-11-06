"""Load the PhysioNet/Computing in Cardiology Challenge 2019 data.

What this module does:
  * Reads the pipe-separated .psv files (one file per patient) for hospital A
    (development) and hospital B (external test).
  * Makes patient IDs globally unique by prefixing the hospital
    (file p000001.psv at hospital A -> "A_p000001"), so A and B can never collide.
  * Validates the data: same columns in every file, no duplicate
    (patient, hour) rows, labels are 0 or 1, no missing hours.
  * Proves hospital A and hospital B share no patients (study_design.md, section 4).
  * Prints a summary, including the onset-time checks the study design asks for
    (how many patients have a left-censored onset or a label that turns back off).

Run from the repo root (paths in the config are relative to where you run it):

    PYTHONPATH=src python -m sepsis_icu.data --config configs/default.yaml

Use --limit 200 for a quick check on the first 200 files per hospital.
The summary only describes the data. Nothing here tunes anything on hospital B.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from sepsis_icu.onset import LABEL_SHIFT_HOURS, reconstruct_onset
from sepsis_icu.splits import assert_disjoint

ID_COLUMN = "patient_id"
HOSPITAL_COLUMN = "hospital"
TIME_COLUMN = "ICULOS"
LABEL_COLUMN = "SepsisLabel"

HOSPITAL_A = "A"  # development
HOSPITAL_B = "B"  # external test

# Columns that describe the patient rather than a time-varying measurement.
# Check these names against the header of your .psv files.
STATIC_COLUMNS = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")


def value_columns(
    df: pd.DataFrame,
    id_col: str = ID_COLUMN,
    time_col: str = TIME_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> List[str]:
    """Time-varying measurement columns (vitals and labs): everything that is not
    an ID, the hospital, the time, the label, or a static patient column."""
    skip = {id_col, HOSPITAL_COLUMN, time_col, label_col, *STATIC_COLUMNS}
    return [c for c in df.columns if c not in skip]


def read_psv(
    path: str,
    hospital: str,
    id_col: str = ID_COLUMN,
    time_col: str = TIME_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Read one patient file and add the hospital-prefixed patient ID."""
    path = Path(path)
    df = pd.read_csv(path, sep="|")
    missing = [c for c in (time_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if id_col in df.columns or HOSPITAL_COLUMN in df.columns:
        raise ValueError(f"{path.name}: already has a '{id_col}' or '{HOSPITAL_COLUMN}' column")
    df.insert(0, id_col, f"{hospital}_{path.stem}")
    df.insert(1, HOSPITAL_COLUMN, hospital)
    return df


def validate(
    df: pd.DataFrame,
    id_col: str = ID_COLUMN,
    time_col: str = TIME_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Check basic integrity and return the data sorted by (patient, hour)."""
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} has missing values")
    if df.duplicated([id_col, time_col]).any():
        raise ValueError("duplicate (patient, hour) rows")
    if not df[label_col].isin([0, 1]).all():
        raise ValueError(f"{label_col} must be 0 or 1 in every row")
    return df.sort_values([id_col, time_col], kind="stable").reset_index(drop=True)


def load_hospital(
    directory: str,
    hospital: str,
    limit: Optional[int] = None,
    id_col: str = ID_COLUMN,
    time_col: str = TIME_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Load every .psv file in `directory` into one validated DataFrame."""
    files = sorted(Path(directory).glob("*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv files found in {directory}")
    if limit is not None:
        files = files[:limit]

    frames = []
    header: Optional[List[str]] = None
    for f in files:
        df = read_psv(str(f), hospital, id_col, time_col, label_col)
        if header is None:
            header = list(df.columns)
        elif list(df.columns) != header:
            raise ValueError(f"{f.name}: columns differ from {files[0].name}")
        frames.append(df)

    return validate(pd.concat(frames, ignore_index=True), id_col, time_col, label_col)


def load_from_config(path: str, limit: Optional[int] = None):
    """Load hospital A and B as set in the config.

    Returns (development_df, external_df, config_dict). Raises if the two
    hospitals share any patient ID.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    d = cfg["data"]
    cols = dict(
        id_col=d.get("patient_id_column", ID_COLUMN),
        time_col=d.get("time_column", TIME_COLUMN),
        label_col=d.get("label_column", LABEL_COLUMN),
    )
    dev = load_hospital(d["development_dir"], HOSPITAL_A, limit=limit, **cols)
    ext = load_hospital(d["external_test_dir"], HOSPITAL_B, limit=limit, **cols)
    assert_disjoint(
        development=dev[cols["id_col"]].unique(),
        hospital_b=ext[cols["id_col"]].unique(),
    )
    return dev, ext, cfg


def summarize(
    df: pd.DataFrame,
    shift: float = LABEL_SHIFT_HOURS,
    id_col: str = ID_COLUMN,
    time_col: str = TIME_COLUMN,
    label_col: str = LABEL_COLUMN,
) -> Dict[str, float]:
    """Counts used to check the labels and onset times (study_design.md, section 3)."""
    septic = left_censored = non_monotone = 0
    for _, g in df.groupby(id_col, sort=False):
        o = reconstruct_onset(g[time_col].to_numpy(), g[label_col].to_numpy(), shift)
        if o.onset_hour is not None:
            septic += 1
            left_censored += int(o.left_censored)
            non_monotone += int(not o.label_monotone)

    stay = df.groupby(id_col, sort=False).size()
    n_patients = int(stay.size)
    return {
        "patients": n_patients,
        "patient_hours (rows)": int(len(df)),
        "septic patients": septic,
        "septic patients (%)": round(100 * septic / n_patients, 2) if n_patients else 0.0,
        "label=1 hours (%)": round(100 * float(df[label_col].mean()), 2) if len(df) else 0.0,
        "left-censored onsets": left_censored,
        "label turns back to 0": non_monotone,
        "median stay (hours)": float(stay.median()) if n_patients else 0.0,
        "longest stay (hours)": int(stay.max()) if n_patients else 0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Load the Challenge data and print a summary.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only load the first N files per hospital (quick check only)",
    )
    args = parser.parse_args(argv)

    dev, ext, cfg = load_from_config(args.config, args.limit)
    shift = cfg.get("labels", {}).get("label_shift_hours", LABEL_SHIFT_HOURS)
    print(f"label shift used for onset: {shift} hours")
    print(f"columns used as measurements: {len(value_columns(dev))}")
    for name, df in (("Hospital A (development)", dev), ("Hospital B (external test)", ext)):
        print(f"\n{name}")
        for key, val in summarize(df, shift).items():
            print(f"  {key:<24} {val}")
    print("\nHospital A and B share no patients: OK")


if __name__ == "__main__":
    main()
