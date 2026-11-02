"""Onset-time reconstruction and lead-time binning.

Everything in the early-vs-late analysis (study_design.md, section 6) depends on
the onset time, so this module is small on purpose and fully unit-tested
(see tests/test_onset.py).

Convention
----------
In the PhysioNet/Computing in Cardiology Challenge 2019 data, SepsisLabel turns
to 1 about 6 hours BEFORE the clinically defined onset (t_sepsis - 6). So:

    onset_hour = first hour with SepsisLabel == 1  +  LABEL_SHIFT_HOURS

VERIFY the 6-hour shift against the Challenge documentation, then record the
verification (date + source) in the decisions log in study_design.md.

Time axis
---------
`hours` is the recorded hour of each row (the ICULOS column), NOT the row
index. Rows must be strictly increasing in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Verify against Challenge documentation, then log it in study_design.md.
LABEL_SHIFT_HOURS: float = 6.0

# Lead-time bins (study_design.md, section 6)
BIN_EARLY = "early"
BIN_INTERMEDIATE = "intermediate"
BIN_LATE = "late"
BIN_MISSED = "missed_or_post_onset"


@dataclass(frozen=True)
class Onset:
    """Result of onset reconstruction for one patient."""

    # None for patients who never become septic.
    onset_hour: Optional[float]
    # Hour of the first SepsisLabel == 1 row (None if never).
    first_label_hour: Optional[float]
    # True if the label is already 1 at the first recorded hour. The label
    # turned on at or before that hour, so the true onset could be earlier
    # than reconstructed. Treat onset_hour as an upper bound for these patients
    # and decide (and log) whether to exclude them from lead-time analysis.
    left_censored: bool
    # False if the label ever goes back to 0 after turning on (unexpected).
    label_monotone: bool


def _as_1d_float(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def reconstruct_onset(
    hours: Sequence[float],
    labels: Sequence[float],
    shift: float = LABEL_SHIFT_HOURS,
) -> Onset:
    """Reconstruct sepsis onset time for one patient.

    Parameters
    ----------
    hours : recorded hour for each row (ICULOS), strictly increasing.
    labels : SepsisLabel for each row, 0 or 1.
    shift : hours between the label turning on and the clinical onset.
    """
    h = _as_1d_float("hours", hours)
    y = _as_1d_float("labels", labels)

    if h.shape != y.shape:
        raise ValueError("hours and labels must have the same length")
    if h.size == 0:
        raise ValueError("empty patient record")
    if np.isnan(h).any() or np.isnan(y).any():
        raise ValueError("hours and labels must not contain NaN")
    if np.any(np.diff(h) <= 0):
        raise ValueError("hours must be strictly increasing")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")

    positives = np.flatnonzero(y == 1.0)
    if positives.size == 0:
        return Onset(
            onset_hour=None,
            first_label_hour=None,
            left_censored=False,
            label_monotone=True,
        )

    first = int(positives[0])
    return Onset(
        onset_hour=float(h[first] + shift),
        first_label_hour=float(h[first]),
        left_censored=(first == 0),
        label_monotone=bool(np.all(y[first:] == 1.0)),
    )


def first_alert_hour(
    hours: Sequence[float],
    scores: Sequence[float],
    threshold: float,
) -> Optional[float]:
    """Hour of the first row whose score is >= threshold, or None if never."""
    h = _as_1d_float("hours", hours)
    s = _as_1d_float("scores", scores)
    if h.shape != s.shape:
        raise ValueError("hours and scores must have the same length")
    idx = np.flatnonzero(s >= threshold)
    return float(h[idx[0]]) if idx.size else None


def assign_lead_time_bin(
    onset_hour: Optional[float],
    alert_hour: Optional[float],
    early_min: float = 12.0,
    intermediate_min: float = 6.0,
) -> Optional[str]:
    """Assign a septic patient to a lead-time bin.

    lead = onset_hour - alert_hour (hours the alert came before onset).

        early         lead >= early_min
        intermediate  intermediate_min <= lead < early_min
        late          0 <= lead < intermediate_min
        missed        lead < 0 (first alert after onset) or never alerted

    Boundary hours are assigned to the earlier bin (lead == 12 -> early,
    lead == 6 -> intermediate, lead == 0 -> late). The study design does not
    say which way boundaries go; log this choice in study_design.md.

    Returns None when onset_hour is None: non-sepsis patients are not binned.
    """
    if onset_hour is None:
        return None
    if alert_hour is None:
        return BIN_MISSED

    lead = onset_hour - alert_hour
    if lead < 0:
        return BIN_MISSED
    if lead >= early_min:
        return BIN_EARLY
    if lead >= intermediate_min:
        return BIN_INTERMEDIATE
    return BIN_LATE
