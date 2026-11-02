"""Unit tests for onset reconstruction and lead-time binning.

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

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


# ---------- reconstruct_onset ----------

def test_default_shift_is_six_hours():
    assert LABEL_SHIFT_HOURS == 6.0


def test_non_sepsis_patient_has_no_onset():
    o = reconstruct_onset([1, 2, 3, 4], [0, 0, 0, 0])
    assert o.onset_hour is None
    assert o.first_label_hour is None
    assert o.left_censored is False
    assert o.label_monotone is True


def test_onset_is_first_label_hour_plus_shift():
    hours = list(range(1, 13))
    labels = [0] * 4 + [1] * 8  # first label at hour 5
    o = reconstruct_onset(hours, labels)
    assert o.first_label_hour == 5
    assert o.onset_hour == 11
    assert o.left_censored is False
    assert o.label_monotone is True


def test_uses_recorded_hours_not_row_index():
    # Record starts at hour 10, so the third row is hour 12, not index 2.
    o = reconstruct_onset([10, 11, 12, 13], [0, 0, 1, 1])
    assert o.first_label_hour == 12
    assert o.onset_hour == 18


def test_custom_shift():
    o = reconstruct_onset([1, 2, 3, 4, 5], [0, 0, 0, 1, 1], shift=0)
    assert o.onset_hour == 4


def test_label_already_on_at_first_row_is_left_censored():
    o = reconstruct_onset([1, 2, 3], [1, 1, 1])
    assert o.left_censored is True
    assert o.onset_hour == 7  # 1 + 6, but only an upper bound


def test_label_off_then_on_is_not_left_censored():
    o = reconstruct_onset([1, 2, 3], [0, 1, 1])
    assert o.left_censored is False


def test_non_monotone_label_is_flagged_and_uses_first_positive():
    o = reconstruct_onset([1, 2, 3, 4], [0, 1, 0, 1])
    assert o.label_monotone is False
    assert o.first_label_hour == 2
    assert o.onset_hour == 8


@pytest.mark.parametrize(
    "hours, labels",
    [
        ([1, 2, 3], [0, 1]),          # length mismatch
        ([], []),                      # empty record
        ([1, 3, 2], [0, 0, 1]),        # decreasing hours
        ([1, 2, 2], [0, 0, 1]),        # duplicate hours
        ([1, 2, 3], [0, 2, 1]),        # non-binary label
        ([1, 2, 3], [0, float("nan"), 1]),  # NaN label
        ([1, float("nan"), 3], [0, 0, 1]),  # NaN hour
    ],
)
def test_invalid_inputs_raise(hours, labels):
    with pytest.raises(ValueError):
        reconstruct_onset(hours, labels)


# ---------- first_alert_hour ----------

def test_first_alert_threshold_is_inclusive():
    assert first_alert_hour([1, 2, 3, 4], [0.1, 0.5, 0.9, 0.9], 0.5) == 2


def test_first_alert_returns_first_crossing_even_if_score_drops_later():
    assert first_alert_hour([1, 2, 3, 4], [0.1, 0.8, 0.2, 0.9], 0.5) == 2


def test_never_alerted_returns_none():
    assert first_alert_hour([1, 2, 3], [0.1, 0.2, 0.3], 0.5) is None


def test_first_alert_length_mismatch_raises():
    with pytest.raises(ValueError):
        first_alert_hour([1, 2, 3], [0.1, 0.2], 0.5)


# ---------- assign_lead_time_bin ----------

@pytest.mark.parametrize(
    "onset, alert, expected",
    [
        (20, 8, BIN_EARLY),            # lead 12 (boundary -> early)
        (20, 1, BIN_EARLY),            # lead 19
        (20, 8.5, BIN_INTERMEDIATE),   # lead 11.5
        (20, 14, BIN_INTERMEDIATE),    # lead 6 (boundary -> intermediate)
        (20, 14.5, BIN_LATE),          # lead 5.5
        (20, 20, BIN_LATE),            # lead 0 (alert at onset -> late)
        (20, 20.5, BIN_MISSED),        # alert after onset
        (20, None, BIN_MISSED),        # never alerted
    ],
)
def test_lead_time_bins(onset, alert, expected):
    assert assign_lead_time_bin(onset, alert) == expected


@pytest.mark.parametrize("alert", [None, 3, 100])
def test_non_sepsis_patients_are_not_binned(alert):
    assert assign_lead_time_bin(None, alert) is None


def test_custom_bin_edges():
    assert assign_lead_time_bin(20, 10, early_min=10, intermediate_min=4) == BIN_EARLY
    assert assign_lead_time_bin(20, 15, early_min=10, intermediate_min=4) == BIN_INTERMEDIATE
    assert assign_lead_time_bin(20, 18, early_min=10, intermediate_min=4) == BIN_LATE


# ---------- end to end ----------

def test_end_to_end_early_alert():
    hours = list(range(1, 31))
    labels = [0] * 12 + [1] * 18          # first label at hour 13 -> onset 19
    scores = [0.1] * 4 + [0.9] * 26       # first alert at hour 5
    onset = reconstruct_onset(hours, labels)
    alert = first_alert_hour(hours, scores, 0.5)
    assert onset.onset_hour == 19
    assert alert == 5
    assert assign_lead_time_bin(onset.onset_hour, alert) == BIN_EARLY  # lead 14
