"""Recalibration (study_design.md, section 7). numpy and pandas only.

Three ways to fix miscalibrated probabilities, each with fit(p, y) and transform(p):

  PlattScaler          logistic recalibration: new logit = intercept + slope * logit(p)
  IsotonicCalibrator   monotone step-like mapping fit with pool-adjacent-violators
  InterceptShift       intercept-only update: new logit = logit(p) + shift

Rules from the study design:
  * Platt scaling and isotonic regression are fit on DEVELOPMENT data only
    (use the out-of-fold predictions from sepsis_icu.models), then applied
    unchanged to the patient-level holdout and to hospital B.
  * For hospital B, ALSO report an intercept-only update fit on a small labeled
    portion of B and evaluated on the rest. split_labeled_portion makes that
    patient-level split; this is the only place any hospital B labels are used
    for fitting, and the evaluation patients are never touched.
  * Never evaluate a calibrator on the same data it was fit on; that looks
    better than it is.

The same classes work at hour level and at patient level (see
sepsis_icu.metrics.patient_level). Fit a separate calibrator for each level.

Platt scaling and the intercept update are monotone, so they never change
AUROC. Isotonic regression is monotone but flat in places, which creates ties
and can lower AUROC slightly. It can also output exactly 0 or 1.

compare_calibration builds the before/after table for the report.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from sepsis_icu.metrics import auroc, brier, calibration_intercept_slope, ece
from sepsis_icu.splits import assert_disjoint, patient_outcomes

_EPS = 1e-6


def logit(p: Sequence[float]) -> np.ndarray:
    """log(p / (1 - p)), with p clipped away from exactly 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def expit(z: Sequence[float]) -> np.ndarray:
    """Inverse of logit: 1 / (1 + exp(-z))."""
    z = np.clip(np.asarray(z, dtype=float), -700, 700)
    return 1.0 / (1.0 + np.exp(-z))


def _prepare_p(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D array")
    if np.isnan(p).any():
        raise ValueError("probabilities contain NaN")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    return p


def _prepare_fit(p: Sequence[float], y: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    p = _prepare_p(p)
    y = np.asarray(y)
    if y.shape != p.shape:
        raise ValueError("p and y must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    y = y.astype(int)
    if y.min() == y.max():
        raise ValueError("calibration needs both outcomes (0 and 1) in the data it is fit on")
    return p, y


class PlattScaler:
    """Logistic recalibration. After fit: intercept_ and slope_.

    slope_ below 1 means the original predictions were too extreme.
    """

    intercept_ = None
    slope_ = None

    def fit(self, p: Sequence[float], y: Sequence) -> "PlattScaler":
        p, y = _prepare_fit(p, y)
        intercept, slope = calibration_intercept_slope(y, p)
        if not (np.isfinite(intercept) and np.isfinite(slope)):
            raise ValueError(
                "Platt scaling could not be fit (no spread in the predictions, "
                "or the outcome is perfectly separated)"
            )
        self.intercept_, self.slope_ = intercept, slope
        return self

    def transform(self, p: Sequence[float]) -> np.ndarray:
        if self.slope_ is None:
            raise RuntimeError("call fit() before transform()")
        return expit(self.intercept_ + self.slope_ * logit(_prepare_p(p)))


class InterceptShift:
    """Intercept-only update: new logit = logit(p) + shift_.

    The shift is the maximum-likelihood value, which makes the average updated
    probability equal the observed event rate in the data it was fit on.
    Use it for a new site: it needs far fewer labeled cases than Platt scaling.
    """

    shift_ = None

    def fit(self, p: Sequence[float], y: Sequence) -> "InterceptShift":
        p, y = _prepare_fit(p, y)
        z = logit(p)
        shift = 0.0
        for _ in range(200):
            q = expit(z + shift)
            gradient = float(np.sum(y - q))
            hessian = float(np.sum(q * (1.0 - q)))
            if hessian < 1e-12:
                raise ValueError("intercept update could not be fit")
            step = float(np.clip(gradient / hessian, -5.0, 5.0))
            shift += step
            if abs(step) < 1e-10:
                self.shift_ = shift
                return self
        raise ValueError("intercept update did not converge")

    def transform(self, p: Sequence[float]) -> np.ndarray:
        if self.shift_ is None:
            raise RuntimeError("call fit() before transform()")
        return expit(logit(_prepare_p(p)) + self.shift_)


class IsotonicCalibrator:
    """Isotonic regression (pool-adjacent-violators). After fit: x_ and y_ knots.

    transform interpolates linearly between knots and clips to the end values
    outside the range seen in fitting. Needs plenty of data; with few cases it
    overfits, so prefer Platt scaling or the intercept update there.
    """

    x_ = None
    y_ = None

    def fit(self, p: Sequence[float], y: Sequence) -> "IsotonicCalibrator":
        p, y = _prepare_fit(p, y)
        order = np.argsort(p, kind="stable")
        xs, ys = p[order], y[order].astype(float)
        unique_x, first = np.unique(xs, return_index=True)
        sum_y = np.add.reduceat(ys, first)
        count = np.diff(np.r_[first, xs.size]).astype(float)

        sums, counts, lows, highs = [], [], [], []
        for x, s, c in zip(unique_x, sum_y, count):
            low = x
            while sums and sums[-1] * c >= s * counts[-1]:  # previous mean >= this mean
                s += sums.pop()
                c += counts.pop()
                low = lows.pop()
                highs.pop()
            sums.append(s)
            counts.append(c)
            lows.append(low)
            highs.append(x)

        values = np.array(sums) / np.array(counts)
        knots_x, knots_y = [], []
        for low, high, value in zip(lows, highs, values):
            knots_x.append(low)
            knots_y.append(value)
            if high > low:
                knots_x.append(high)
                knots_y.append(value)
        self.x_, self.y_ = np.array(knots_x), np.array(knots_y)
        return self

    def transform(self, p: Sequence[float]) -> np.ndarray:
        if self.x_ is None:
            raise RuntimeError("call fit() before transform()")
        return np.interp(_prepare_p(p), self.x_, self.y_)


def split_labeled_portion(
    patient_ids: Sequence,
    labels: Sequence,
    labeled_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a new site's patients into a small labeled portion and the rest.

    Patient-level and stratified by ever-septic status, so both parts contain
    septic and non-septic patients. Fit the intercept update on the labeled
    portion's rows and evaluate on the evaluation patients' rows.

    Returns (labeled_patient_ids, evaluation_patient_ids).
    """
    if not 0.0 < labeled_fraction < 1.0:
        raise ValueError("labeled_fraction must be between 0 and 1 (exclusive)")
    ids, ever = patient_outcomes(patient_ids, labels)

    rng = np.random.default_rng(seed)
    labeled = np.zeros(ids.size, dtype=bool)
    for stratum in (True, False):
        idx = np.flatnonzero(ever == stratum)
        if idx.size == 0:
            continue
        n_labeled = int(round(labeled_fraction * idx.size))
        if n_labeled < 1 or n_labeled >= idx.size:
            name = "septic" if stratum else "non-septic"
            raise ValueError(
                f"labeled_fraction={labeled_fraction} leaves no {name} patients on one side "
                f"({idx.size} {name} patients in total)"
            )
        rng.shuffle(idx)
        labeled[idx[:n_labeled]] = True

    labeled_ids, evaluation_ids = ids[labeled], ids[~labeled]
    assert_disjoint(labeled=labeled_ids, evaluation=evaluation_ids)
    return labeled_ids, evaluation_ids


def compare_calibration(
    y: Sequence,
    predictions: Dict[str, Sequence[float]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Before/after table: one row per variant of the predictions.

    Example:
        compare_calibration(y, {"uncalibrated": p, "platt": platt.transform(p)})

    Columns: variant, auroc, brier, calibration_intercept, calibration_slope,
    ece, mean_predicted, observed_rate. Evaluate on data the calibrators were
    NOT fit on.
    """
    y = np.asarray(y)
    rows = []
    for name, p in predictions.items():
        p = _prepare_p(p)
        if y.shape != p.shape:
            raise ValueError(f"'{name}': predictions and y must have the same length")
        intercept, slope = calibration_intercept_slope(y, p)
        rows.append(
            {
                "variant": name,
                "auroc": auroc(y, p),
                "brier": brier(y, p),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece": ece(y, p, n_bins),
                "mean_predicted": float(p.mean()),
                "observed_rate": float(y.mean()),
            }
        )
    return pd.DataFrame(rows)
