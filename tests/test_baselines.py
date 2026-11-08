"""Unit tests for the baseline scores (SIRS, qSOFA, NEWS2-style, ICU-LOS).

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
from pandas.testing import assert_frame_equal

from sepsis_icu.baselines import (
    CONVENTIONAL_THRESHOLDS,
    baseline_scores,
    icu_los_score,
    latest_values,
    news2_style_score,
    qsofa_score,
    sirs_score,
)

NAN = np.nan
NORMAL = dict(HR=70.0, Temp=37.0, Resp=16.0, SBP=120.0, O2Sat=98.0, WBC=8.0, PaCO2=40.0)


def row(**overrides):
    """One patient-hour of 'latest values'; normal unless overridden."""
    values = {**NORMAL, **overrides}
    return pd.DataFrame({"patient_id": ["p"], "ICULOS": [1], **{k: [v] for k, v in values.items()}})


def score(fn, **overrides):
    return float(fn(row(**overrides))[0])


# ---------- a normal patient scores zero everywhere ----------

def test_normal_values_score_zero():
    r = row()
    assert sirs_score(r)[0] == 0
    assert qsofa_score(r)[0] == 0
    assert news2_style_score(r)[0] == 0


# ---------- SIRS ----------

@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(Temp=38.5), 1),
        (dict(Temp=38.0), 0),   # criterion is > 38
        (dict(Temp=35.9), 1),
        (dict(Temp=36.0), 0),   # criterion is < 36
        (dict(HR=91), 1),
        (dict(HR=90), 0),
        (dict(Resp=21), 1),
        (dict(Resp=20), 0),
        (dict(PaCO2=31), 1),    # PaCO2 < 32 also meets the respiratory criterion
        (dict(PaCO2=32), 0),
        (dict(WBC=12.1), 1),
        (dict(WBC=12.0), 0),
        (dict(WBC=3.9), 1),
        (dict(WBC=4.0), 0),
    ],
)
def test_sirs_single_criteria_and_boundaries(overrides, expected):
    assert score(sirs_score, **overrides) == expected


def test_sirs_respiratory_criterion_counts_once_even_if_both_parts_met():
    assert score(sirs_score, Resp=25, PaCO2=25) == 1


def test_sirs_all_four_criteria():
    assert score(sirs_score, Temp=39, HR=110, Resp=25, WBC=15) == 4


# ---------- qSOFA ----------

@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(Resp=22), 1),
        (dict(Resp=21), 0),
        (dict(SBP=100), 1),
        (dict(SBP=101), 0),
        (dict(Resp=30, SBP=80), 2),
    ],
)
def test_qsofa(overrides, expected):
    assert score(qsofa_score, **overrides) == expected


# ---------- NEWS2-style bands (every published boundary) ----------

@pytest.mark.parametrize(
    "resp, pts",
    [(8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3)],
)
def test_news2_respiratory_rate_bands(resp, pts):
    assert score(news2_style_score, Resp=resp) == pts


@pytest.mark.parametrize(
    "spo2, pts",
    [(91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0)],
)
def test_news2_spo2_bands(spo2, pts):
    assert score(news2_style_score, O2Sat=spo2) == pts


@pytest.mark.parametrize(
    "sbp, pts",
    [(90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3)],
)
def test_news2_systolic_bp_bands(sbp, pts):
    assert score(news2_style_score, SBP=sbp) == pts


@pytest.mark.parametrize(
    "hr, pts",
    [(40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (110, 1), (111, 2), (130, 2), (131, 3)],
)
def test_news2_pulse_bands(hr, pts):
    assert score(news2_style_score, HR=hr) == pts


@pytest.mark.parametrize(
    "temp, pts",
    [(35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0), (38.1, 1), (39.0, 1), (39.1, 2)],
)
def test_news2_temperature_bands(temp, pts):
    assert score(news2_style_score, Temp=temp) == pts


def test_news2_components_add_up():
    # Resp 22 (2) + SpO2 93 (2) + SBP 95 (2) + HR 115 (2) + Temp 39.5 (2) = 10
    assert score(
        news2_style_score, Resp=22, O2Sat=93, SBP=95, HR=115, Temp=39.5
    ) == 10


def test_news2_maximum_is_fifteen():
    assert score(
        news2_style_score, Resp=5, O2Sat=85, SBP=70, HR=150, Temp=34
    ) == 15


# ---------- missing values ----------

def test_missing_values_score_zero():
    r = row(HR=NAN, Temp=NAN, Resp=NAN, SBP=NAN, O2Sat=NAN, WBC=NAN, PaCO2=NAN)
    assert sirs_score(r)[0] == 0
    assert qsofa_score(r)[0] == 0
    assert news2_style_score(r)[0] == 0


# ---------- latest values: forward-fill, causality ----------

def make_patient(pid, hr_values, **others):
    n = len(hr_values)
    data = {"patient_id": pid, "ICULOS": np.arange(1, n + 1), "HR": hr_values}
    for name, default in NORMAL.items():
        if name != "HR":
            data[name] = others.get(name, [default] * n)
    return pd.DataFrame(data)


def test_last_known_value_persists_until_remeasured():
    df = make_patient("p1", [120, NAN, NAN, 70, NAN])
    out = baseline_scores(df)
    # HR 120 is carried forward (SIRS HR>90 met) until HR 70 is measured at hour 4.
    assert out["sirs"].tolist() == [1, 1, 1, 0, 0]


def test_values_are_not_carried_across_patients():
    df = pd.concat(
        [make_patient("p1", [120, 120, 120]), make_patient("p2", [NAN, NAN, NAN])],
        ignore_index=True,
    )
    out = baseline_scores(df)
    p2 = out[out["patient_id"] == "p2"]
    assert p2["sirs"].tolist() == [0, 0, 0]


def test_scores_at_t_ignore_the_future():
    rng = np.random.default_rng(0)
    frames = []
    for i in range(6):
        n = 20
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": f"p{i}",
                    "ICULOS": np.arange(1, n + 1),
                    **{
                        name: np.where(rng.random(n) < 0.5, np.nan, rng.normal(mu, sd, n))
                        for name, (mu, sd) in dict(
                            HR=(85, 20), Temp=(37, 1), Resp=(18, 5), SBP=(115, 20),
                            O2Sat=(95, 4), WBC=(10, 4), PaCO2=(38, 6),
                        ).items()
                    },
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    cutoff = 10

    full = baseline_scores(df)
    full_before = full[full["ICULOS"] <= cutoff].reset_index(drop=True)
    truncated = baseline_scores(df[df["ICULOS"] <= cutoff]).reset_index(drop=True)
    assert_frame_equal(full_before, truncated)

    junk = df.copy()
    future = junk["ICULOS"] > cutoff
    for name in ["HR", "Temp", "Resp", "SBP", "O2Sat", "WBC", "PaCO2"]:
        junk.loc[future, name] = 999.0
    perturbed = baseline_scores(junk)
    perturbed_before = perturbed[perturbed["ICULOS"] <= cutoff].reset_index(drop=True)
    assert_frame_equal(full_before, perturbed_before)


# ---------- icu_los and output shape ----------

def test_icu_los_score_is_hours_since_admission():
    latest = latest_values(make_patient("p1", [80, 80, 80, 80]))
    assert icu_los_score(latest).tolist() == [1, 2, 3, 4]


def test_baseline_scores_columns_and_sorting():
    df = pd.concat(
        [make_patient("p2", [80, 80]), make_patient("p1", [80, 80, 80])], ignore_index=True
    ).sample(frac=1, random_state=1)
    out = baseline_scores(df)
    assert list(out.columns) == [
        "patient_id", "ICULOS", "icu_los", "sirs", "qsofa", "news2_style"
    ]
    keys = out[["patient_id", "ICULOS"]]
    assert keys.equals(keys.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True))
    assert len(out) == 5


def test_missing_required_columns_raise():
    df = make_patient("p1", [80, 80]).drop(columns=["WBC"])
    with pytest.raises(ValueError, match="WBC"):
        baseline_scores(df)


def test_conventional_thresholds():
    assert CONVENTIONAL_THRESHOLDS == {"sirs": 2, "qsofa": 2, "news2_style": 5}
