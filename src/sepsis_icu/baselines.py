"""Simple baselines the model must beat (study_design.md, section 8).

Scores computed here, one value per patient-hour:

  icu_los       hours since ICU admission (the ICULOS column). If this scores
                suspiciously well, the label or splits are leaking time
                information (study_design.md, section 5).
  sirs          number of SIRS criteria met, 0 to 4
  qsofa         number of qSOFA criteria met, 0 to 2 (see limits below)
  news2_style   NEWS2-style points, 0 to 15 (see limits below)

All scores use the LATEST KNOWN value of each variable (forward-filled within
each patient), so a score at hour t only uses hours <= t. A variable that has
never been measured yet scores 0 (treated as normal).

Limits, because the Challenge data lacks some inputs. Record these in the
decisions log in study_design.md:
  * qSOFA normally also counts altered mentation (GCS < 15). There is no GCS
    in the data, so only respiratory rate and systolic BP are counted. The
    maximum is 2, so the conventional alert (>= 2) needs both.
  * NEWS2 normally also counts supplemental oxygen and level of consciousness.
    Neither is available, so only respiration rate, SpO2 (scale 1), systolic
    BP, pulse and temperature are scored (maximum 15, not 20).
  * SIRS normally allows "bands > 10%" for the white cell criterion. Not
    available here.

Band edges (NEWS2): a value is compared with the upper edge of each published
band, so a value between two bands (for example a respiratory rate of 11.5)
falls in the higher band.

Units follow the Challenge data: Temp in degrees C, HR in beats/min,
Resp in breaths/min, SBP in mmHg, O2Sat in %, WBC in 10^3/uL, PaCO2 in mmHg.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from sepsis_icu.features import forward_fill

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_TIME_COLUMN = "ICULOS"

REQUIRED_COLUMNS = ("HR", "Temp", "Resp", "SBP", "O2Sat", "WBC", "PaCO2")

# Conventional alert thresholds (score >= threshold). For comparisons at a fixed
# alert rate (study_design.md, section 7), use sepsis_icu.metrics instead.
CONVENTIONAL_THRESHOLDS: Dict[str, float] = {"sirs": 2, "qsofa": 2, "news2_style": 5}


def latest_values(
    df: pd.DataFrame,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    """Latest known value of each required variable, per patient-hour.

    Returns a DataFrame sorted by (patient, time) with columns id_col, time_col
    and the REQUIRED_COLUMNS, forward-filled within each patient.
    """
    missing = [c for c in (id_col, time_col, *REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns needed for baseline scores: {missing}")
    df = df.sort_values([id_col, time_col], kind="stable").reset_index(drop=True)
    latest = forward_fill(df, list(REQUIRED_COLUMNS), id_col, time_col)
    latest.insert(0, time_col, df[time_col])
    latest.insert(0, id_col, df[id_col])
    return latest


def _col(latest: pd.DataFrame, name: str) -> np.ndarray:
    return np.asarray(latest[name], dtype=float)


def _band_points(values: np.ndarray, edges: Sequence[float], points: Sequence[float]) -> np.ndarray:
    """points[i] for the first band whose upper edge is >= value; points[-1] above
    the last edge. Missing values score 0."""
    out = np.full(values.shape, points[-1], dtype=float)
    for edge, pts in reversed(list(zip(edges, points))):
        out = np.where(values <= edge, pts, out)
    return np.where(np.isnan(values), 0.0, out)


def icu_los_score(latest: pd.DataFrame, time_col: str = DEFAULT_TIME_COLUMN) -> np.ndarray:
    """Hours since ICU admission, as a risk score."""
    return np.asarray(latest[time_col], dtype=float)


def sirs_score(latest: pd.DataFrame) -> np.ndarray:
    """SIRS criteria met (0 to 4).

    Temp > 38 or < 36 C; HR > 90; Resp > 20 or PaCO2 < 32; WBC > 12 or < 4.
    Missing values do not meet a criterion.
    """
    temp, hr = _col(latest, "Temp"), _col(latest, "HR")
    resp, paco2, wbc = _col(latest, "Resp"), _col(latest, "PaCO2"), _col(latest, "WBC")
    return (
        ((temp > 38) | (temp < 36)).astype(float)
        + (hr > 90).astype(float)
        + ((resp > 20) | (paco2 < 32)).astype(float)
        + ((wbc > 12) | (wbc < 4)).astype(float)
    )


def qsofa_score(latest: pd.DataFrame) -> np.ndarray:
    """qSOFA criteria met, 0 to 2 (altered mentation is not available).

    Resp >= 22; SBP <= 100.
    """
    resp, sbp = _col(latest, "Resp"), _col(latest, "SBP")
    return (resp >= 22).astype(float) + (sbp <= 100).astype(float)


def news2_style_score(latest: pd.DataFrame) -> np.ndarray:
    """NEWS2-style points from the five available components (0 to 15)."""
    resp = _band_points(_col(latest, "Resp"), [8, 11, 20, 24], [3, 1, 0, 2, 3])
    spo2 = _band_points(_col(latest, "O2Sat"), [91, 93, 95], [3, 2, 1, 0])
    sbp = _band_points(_col(latest, "SBP"), [90, 100, 110, 219], [3, 2, 1, 0, 3])
    pulse = _band_points(_col(latest, "HR"), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    temp = _band_points(_col(latest, "Temp"), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    return resp + spo2 + sbp + pulse + temp


def baseline_scores(
    df: pd.DataFrame,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    """All baseline scores for every patient-hour.

    Returns a DataFrame sorted by (patient, time) with columns:
    id_col, time_col, icu_los, sirs, qsofa, news2_style.
    Labels are not included; join them back on (id_col, time_col).
    """
    latest = latest_values(df, id_col, time_col)
    return pd.DataFrame(
        {
            id_col: latest[id_col],
            time_col: latest[time_col],
            "icu_los": icu_los_score(latest, time_col),
            "sirs": sirs_score(latest),
            "qsofa": qsofa_score(latest),
            "news2_style": news2_style_score(latest),
        }
    )
