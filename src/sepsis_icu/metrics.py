"""Metrics (study_design.md, section 7). numpy and pandas only.

Everything works on row-level arrays, one entry per patient-hour:
  y       0/1 label for that hour
  scores  the model score or baseline score for that hour

Included:
  Discrimination  auroc, auprc (and prevalence, which must be stated with AUPRC)
  Calibration     brier, calibration intercept and slope, ECE, reliability table
  Operating point threshold_for_alert_rate, operating_point (sensitivity, PPV,
                  alerts per patient-day)
  Uncertainty     bootstrap_ci: 95% percentile CIs, resampling PATIENTS

Definitions to log in the decisions log (study_design.md):
  * An "alert" is any patient-hour with score >= threshold. There is no
    suppression after the first alert, so alerts per patient-day counts every
    alerting hour: (alerting hours) / (patient-hours / 24).
  * Sensitivity and PPV here are hour-level. Patient-level detection and lead
    time are handled by the lead-time bins in sepsis_icu.onset.
  * The threshold is chosen on development data only (threshold_for_alert_rate)
    and then carried unchanged to the holdout and to hospital B.
  * Calibration slope and intercept come from a logistic regression of the
    outcome on logit(predicted probability); well calibrated is slope 1,
    intercept 0.
  * ECE depends on the binning, so treat it as a secondary metric.
  * Calibration metrics only make sense for probabilities. For baseline scores
    such as SIRS, call evaluate(..., is_probability=False).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0
_EPS = 1e-6

METRIC_NAMES = (
    "prevalence",
    "auroc",
    "auprc",
    "brier",
    "calibration_intercept",
    "calibration_slope",
    "ece",
    "sensitivity",
    "ppv",
    "alerts_per_patient_day",
)


def _check(y: Sequence, scores: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y and scores must be 1-D arrays of the same length")
    if y.size == 0:
        raise ValueError("no rows given")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y.astype(int), s


def _check_probabilities(p: np.ndarray) -> None:
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be between 0 and 1")


# ---------- discrimination ----------

def prevalence(y: Sequence) -> float:
    """Fraction of rows with label 1. State this next to AUPRC."""
    y, _ = _check(y, np.zeros(len(y)))
    return float(y.mean())


def auroc(y: Sequence, scores: Sequence) -> float:
    """Area under the ROC curve (ties count half). NaN if only one class is present."""
    y, s = _check(y, scores)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: Sequence, scores: Sequence) -> float:
    """Area under the precision-recall curve (average precision). NaN if no positives."""
    y, s = _check(y, scores)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    last_of_each_score = np.r_[np.flatnonzero(np.diff(s_sorted)), y_sorted.size - 1]
    tp, fp = tp[last_of_each_score], fp[last_of_each_score]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


# ---------- calibration ----------

def brier(y: Sequence, p: Sequence) -> float:
    """Mean squared difference between predicted probability and outcome."""
    y, p = _check(y, p)
    _check_probabilities(p)
    return float(np.mean((p - y) ** 2))


def _logistic_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8):
    """Intercept and slope of a logistic regression of y on x (Newton-Raphson).
    Returns (nan, nan) if it cannot be fit (no spread in x, perfect separation)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            eta = np.clip(X @ beta, -35, 35)
            prob = 1.0 / (1.0 + np.exp(-eta))
            weights = prob * (1.0 - prob)
            gradient = X.T @ (y - prob)
            hessian = X.T @ (X * weights[:, None])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return float("nan"), float("nan")
            beta = beta + step
            if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e3:
                return float("nan"), float("nan")
            if np.max(np.abs(step)) < tol:
                return float(beta[0]), float(beta[1])
    return float("nan"), float("nan")


def calibration_intercept_slope(y: Sequence, p: Sequence) -> Tuple[float, float]:
    """(intercept, slope) from regressing the outcome on logit(p).

    Slope < 1 means predictions are too extreme (overconfident); a negative
    intercept means predictions are too high on average.
    """
    y, p = _check(y, p)
    _check_probabilities(p)
    p = np.clip(p, _EPS, 1 - _EPS)
    return _logistic_fit(np.log(p / (1 - p)), y.astype(float))


def reliability_table(y: Sequence, p: Sequence, n_bins: int = 10) -> pd.DataFrame:
    """Data for a reliability diagram: equal-width probability bins.

    Columns: bin_low, bin_high, n, mean_predicted, observed_rate
    (mean_predicted and observed_rate are NaN for empty bins).
    """
    y, p = _check(y, p)
    _check_probabilities(p)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    bin_index = np.clip(np.floor(p * n_bins), 0, n_bins - 1).astype(int)
    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        n = int(in_bin.sum())
        rows.append(
            {
                "bin_low": b / n_bins,
                "bin_high": (b + 1) / n_bins,
                "n": n,
                "mean_predicted": float(p[in_bin].mean()) if n else float("nan"),
                "observed_rate": float(y[in_bin].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ece(y: Sequence, p: Sequence, n_bins: int = 10) -> float:
    """Expected calibration error: bin-weighted mean |observed - predicted|.
    Depends on the binning; report as a secondary metric only."""
    table = reliability_table(y, p, n_bins)
    used = table[table["n"] > 0]
    weights = used["n"] / used["n"].sum()
    return float((weights * (used["observed_rate"] - used["mean_predicted"]).abs()).sum())


# ---------- operating point ----------

def threshold_for_alert_rate(scores: Sequence, alerts_per_patient_day: float) -> float:
    """Lowest threshold whose alert rate does not exceed the target.

    Alert rate = (rows with score >= threshold) / (rows / 24). Call this on
    DEVELOPMENT data only, then reuse the threshold on the holdout and on
    hospital B. Returns +inf if the target allows no alerts.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if alerts_per_patient_day < 0:
        raise ValueError("alerts_per_patient_day must be non-negative")

    allowed = int(np.floor(alerts_per_patient_day * s.size / HOURS_PER_DAY))
    if allowed < 1:
        return float("inf")
    if allowed >= s.size:
        return float(s.min())

    threshold = float(np.sort(s)[::-1][allowed - 1])
    if np.sum(s >= threshold) > allowed:  # ties push the count over: step up
        higher = s[s > threshold]
        return float(higher.min()) if higher.size else float("inf")
    return threshold


def operating_point(y: Sequence, scores: Sequence, threshold: float) -> Dict[str, float]:
    """Hour-level sensitivity, PPV and alerts per patient-day at a fixed threshold."""
    y, s = _check(y, scores)
    alert = s >= threshold
    tp = int(np.sum(alert & (y == 1)))
    n_alerts = int(alert.sum())
    n_pos = int(y.sum())
    return {
        "sensitivity": tp / n_pos if n_pos else float("nan"),
        "ppv": tp / n_alerts if n_alerts else float("nan"),
        "alerts_per_patient_day": n_alerts / (y.size / HOURS_PER_DAY),
    }


# ---------- patient-level view ----------

def patient_level(
    patient_ids: Sequence, y: Sequence, scores: Sequence
) -> Tuple[np.ndarray, np.ndarray]:
    """One row per patient: (ever_septic, highest score during the stay).

    Use for patient-level calibration (study_design.md, section 7). Using the
    maximum score is a choice: log it, and recalibrate the patient-level score
    separately because a maximum is not a calibrated probability.
    """
    ids = np.asarray(patient_ids)
    y, s = _check(y, scores)
    if ids.shape != y.shape:
        raise ValueError("patient_ids must have one entry per row")
    unique, inverse = np.unique(ids, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    ever = np.zeros(unique.size, dtype=int)
    ever[inverse[y == 1]] = 1
    highest = np.full(unique.size, -np.inf)
    np.maximum.at(highest, inverse, s)
    return ever, highest


# ---------- everything at once ----------

def evaluate(
    y: Sequence,
    scores: Sequence,
    threshold: Optional[float] = None,
    is_probability: bool = True,
    n_bins: int = 10,
) -> Dict[str, float]:
    """All metrics in one dict (keys in METRIC_NAMES).

    threshold: fixed alert threshold (chosen on development data). If None,
        the operating-point metrics are NaN.
    is_probability: False for scores that are not probabilities (SIRS, qSOFA,
        NEWS2, ICU-LOS); the calibration metrics are then NaN.
    """
    y, s = _check(y, scores)
    nan = float("nan")
    out = {name: nan for name in METRIC_NAMES}
    out["prevalence"] = float(y.mean())
    out["auroc"] = auroc(y, s)
    out["auprc"] = auprc(y, s)

    if is_probability:
        _check_probabilities(s)
        out["brier"] = brier(y, s)
        out["calibration_intercept"], out["calibration_slope"] = calibration_intercept_slope(y, s)
        out["ece"] = ece(y, s, n_bins)

    if threshold is not None:
        out.update(operating_point(y, s, threshold))
    return out


# ---------- patient-level bootstrap ----------

def bootstrap_ci(
    patient_ids: Sequence,
    y: Sequence,
    scores: Sequence,
    metric_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Percentile bootstrap CIs, resampling PATIENTS (all of a patient's rows
    together), never single rows.

    metric_fn(y, scores) -> dict of metric name -> value, for example
        lambda y, s: evaluate(y, s, threshold=thr)
    Keep the threshold fixed inside metric_fn; do not re-derive it per resample.

    Returns columns: metric, estimate (full data), ci_lower, ci_upper,
    n_valid (resamples where the metric could be computed).
    """
    ids = np.asarray(patient_ids)
    y, s = _check(y, scores)
    if ids.shape != y.shape:
        raise ValueError("patient_ids must have one entry per row")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1 (exclusive)")

    _, codes = np.unique(ids, return_inverse=True)
    codes = np.asarray(codes).reshape(-1)
    n_patients = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    lengths = np.bincount(codes, minlength=n_patients)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]

    def rows_for(draw: np.ndarray) -> np.ndarray:
        chosen = lengths[draw]
        total = int(chosen.sum())
        out_start = np.cumsum(chosen) - chosen
        position = np.arange(total) - np.repeat(out_start, chosen) + np.repeat(starts[draw], chosen)
        return order[position]

    estimate = metric_fn(y, s)
    rng = np.random.default_rng(seed)
    draws = {name: np.full(n_resamples, np.nan) for name in estimate}
    for i in range(n_resamples):
        rows = rows_for(rng.integers(0, n_patients, size=n_patients))
        result = metric_fn(y[rows], s[rows])
        for name in estimate:
            draws[name][i] = result.get(name, np.nan)

    alpha = (1 - ci_level) / 2 * 100
    records = []
    for name, value in estimate.items():
        valid = draws[name][~np.isnan(draws[name])]
        if valid.size:
            lower, upper = np.percentile(valid, [alpha, 100 - alpha])
        else:
            lower = upper = float("nan")
        records.append(
            {
                "metric": name,
                "estimate": value,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_valid": int(valid.size),
            }
        )
    return pd.DataFrame(records)
