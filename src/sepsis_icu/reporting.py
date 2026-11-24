"""Tables that describe the cohorts, and saved model files (study_design.md, section 10).

Tables (all numeric, long format, ready to turn into paper tables):
  analysis_counts         patients, septic patients, hours and positive hours in every analysis:
                          hospital A overall, development, holdout, each fold's training and
                          validation part, and hospital B
  characteristics_table   age, stay length, missing data, sex, unit and sepsis for each dataset
  characteristics_shift   hospital A vs hospital B: means or proportions, and the standardized
                          difference of each
  feature_inventory       every feature each model used: variable, kind, window, missingness or not

Saved files:
  save_model / load_model             fitted scikit-learn model (joblib) plus a .json file with the
                                      feature columns, threshold, versions and other details
  save_calibrator / load_calibrator   Platt, isotonic or intercept-shift calibrator as plain JSON

Joblib files are Python pickles. Only load model files that you created yourself.
A model saved with one scikit-learn version may not load, or may behave differently, in another;
load_model warns if the version has changed.

The patient tables come from sepsis_icu.errors.patient_table.
"""

from __future__ import annotations

import json
import math
import platform
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sepsis_icu.recalibration import InterceptShift, IsotonicCalibrator, PlattScaler
from sepsis_icu.splits import PatientSplit, rows_in

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_LABEL_COLUMN = "SepsisLabel"

NUMERIC_VARIABLES = ("age", "stay_hours", "missing_fraction")
CATEGORICAL_LEVELS = {"sex": ("female", "male", "unknown"), "unit": ("MICU", "SICU", "unknown")}


# ---------- counts per analysis ----------

def _count_row(name: str, df: pd.DataFrame, y: np.ndarray, mask: np.ndarray, id_col: str, label_col: str) -> dict:
    return {
        "analysis": name,
        "patients": int(df.loc[mask, id_col].nunique()),
        "septic_patients": int(df.loc[mask & (df[label_col] == 1).to_numpy(), id_col].nunique()),
        "hours": int(mask.sum()),
        "positive_hours": int(np.asarray(y)[mask].sum()),
    }


def analysis_counts(
    df: pd.DataFrame,
    y: Sequence,
    split: PatientSplit,
    external: Optional[Tuple[pd.DataFrame, Sequence]] = None,
    id_col: str = DEFAULT_ID_COLUMN,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """How many patients and events went into every analysis.

    df, y: hospital A rows and the row-level target. split: the patient-level split.
    external: optional (hospital B rows, its target).
    septic_patients counts patients with any SepsisLabel of 1; positive_hours counts target hours.
    Columns: analysis, patients, septic_patients, hours, positive_hours.
    """
    y = np.asarray(y)
    if len(y) != len(df):
        raise ValueError("y must have one entry per row of df")
    pids = df[id_col].to_numpy()

    rows = [
        _count_row("hospital_a_all", df, y, np.ones(len(df), dtype=bool), id_col, label_col),
        _count_row("hospital_a_development", df, y, rows_in(pids, split.development_ids()), id_col, label_col),
        _count_row("hospital_a_holdout", df, y, rows_in(pids, split.holdout_ids()), id_col, label_col),
    ]
    for k in range(split.n_splits):
        train_ids, val_ids = split.train_val_ids(k)
        rows.append(_count_row(f"fold_{k}_training", df, y, rows_in(pids, train_ids), id_col, label_col))
        rows.append(_count_row(f"fold_{k}_validation", df, y, rows_in(pids, val_ids), id_col, label_col))
    if external is not None:
        ext_df, ext_y = external
        if len(ext_y) != len(ext_df):
            raise ValueError("the external target must have one entry per row")
        rows.append(_count_row("hospital_b", ext_df, np.asarray(ext_y), np.ones(len(ext_df), dtype=bool), id_col, label_col))
    return pd.DataFrame(rows)


# ---------- patient characteristics ----------

def _levels(name: str, *tables: pd.DataFrame) -> List[str]:
    base = list(CATEGORICAL_LEVELS[name])
    extra = sorted({str(v) for t in tables for v in t[name].dropna().unique()} - set(base))
    return base + extra


def characteristics_table(patients_by_dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Patient characteristics for each dataset.

    Input: dataset name -> output of sepsis_icu.errors.patient_table.
    Columns: dataset, variable, level, n, value, q25, q75.
      patients                    n = number of patients
      age, stay_hours,
      missing_fraction            n = patients with a value; value = median; q25 and q75 = quartiles
      sex, unit                   one row per level; n = patients; value = percent of patients
      ever_septic                 level "yes"; n = septic patients; value = percent of patients
    """
    tables = list(patients_by_dataset.values())
    nan = float("nan")
    rows = []
    for name, pt in patients_by_dataset.items():
        n_patients = len(pt)
        rows.append({"dataset": name, "variable": "patients", "level": "", "n": n_patients, "value": nan, "q25": nan, "q75": nan})
        for var in NUMERIC_VARIABLES:
            x = pt[var].dropna()
            rows.append(
                {
                    "dataset": name, "variable": var, "level": "", "n": int(len(x)),
                    "value": float(x.median()) if len(x) else nan,
                    "q25": float(x.quantile(0.25)) if len(x) else nan,
                    "q75": float(x.quantile(0.75)) if len(x) else nan,
                }
            )
        for var in CATEGORICAL_LEVELS:
            for level in _levels(var, *tables):
                count = int((pt[var].astype(str) == level).sum())
                rows.append(
                    {
                        "dataset": name, "variable": var, "level": level, "n": count,
                        "value": 100 * count / n_patients if n_patients else nan, "q25": nan, "q75": nan,
                    }
                )
        septic = int(pt["ever_septic"].astype(bool).sum())
        rows.append(
            {
                "dataset": name, "variable": "ever_septic", "level": "yes", "n": septic,
                "value": 100 * septic / n_patients if n_patients else nan, "q25": nan, "q75": nan,
            }
        )
    return pd.DataFrame(rows)


def _smd_from_moments(mean_a: float, mean_b: float, var_a: float, var_b: float) -> float:
    pooled = math.sqrt((var_a + var_b) / 2) if not (math.isnan(var_a) or math.isnan(var_b)) else float("nan")
    return (mean_a - mean_b) / pooled if pooled and not math.isnan(pooled) else float("nan")


def characteristics_shift(
    patients_a: pd.DataFrame,
    patients_b: pd.DataFrame,
    label_a: str = "hospital_a",
    label_b: str = "hospital_b",
) -> pd.DataFrame:
    """Compare two datasets: how different are the patients?

    Columns: variable, level, <label_a>, <label_b>, standardized_difference.
    Numeric variables (age, stay_hours, missing_fraction) give means. Categorical variables
    and ever_septic give the share of patients (0 to 1) at each level. The standardized
    difference is (a - b) divided by the pooled standard deviation; about 0.1 is small,
    0.5 medium, 0.8 large. NaN when it cannot be computed.
    """
    rows = []
    for var in NUMERIC_VARIABLES:
        xa, xb = patients_a[var].dropna(), patients_b[var].dropna()
        mean_a = float(xa.mean()) if len(xa) else float("nan")
        mean_b = float(xb.mean()) if len(xb) else float("nan")
        var_a = float(xa.var()) if len(xa) > 1 else float("nan")
        var_b = float(xb.var()) if len(xb) > 1 else float("nan")
        rows.append(
            {"variable": var, "level": "", label_a: mean_a, label_b: mean_b,
             "standardized_difference": _smd_from_moments(mean_a, mean_b, var_a, var_b)}
        )

    def proportion_row(variable: str, level: str, flags_a: pd.Series, flags_b: pd.Series) -> dict:
        pa = float(flags_a.mean()) if len(flags_a) else float("nan")
        pb = float(flags_b.mean()) if len(flags_b) else float("nan")
        return {
            "variable": variable, "level": level, label_a: pa, label_b: pb,
            "standardized_difference": _smd_from_moments(pa, pb, pa * (1 - pa), pb * (1 - pb)),
        }

    for var in CATEGORICAL_LEVELS:
        for level in _levels(var, patients_a, patients_b):
            rows.append(proportion_row(var, level, patients_a[var].astype(str) == level, patients_b[var].astype(str) == level))
    rows.append(proportion_row("ever_septic", "yes", patients_a["ever_septic"].astype(bool), patients_b["ever_septic"].astype(bool)))
    return pd.DataFrame(rows)


# ---------- which features each model used ----------

_FEATURE_PATTERNS = (
    (re.compile(r"^(?P<var>.+)_hours_since$"), "hours since last measured", False, True),
    (re.compile(r"^(?P<var>.+)_observed$"), "measured this hour", False, True),
    (re.compile(r"^(?P<var>.+)_ffill$"), "latest value", False, False),
    (re.compile(r"^(?P<var>.+)_(?P<stat>mean|min|max|delta)(?P<window>\d+)h$"), None, True, False),
)
_STAT_NAMES = {"mean": "mean over window", "min": "minimum over window", "max": "maximum over window", "delta": "change over window"}


def feature_inventory(columns_by_model: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """List every feature each model used, split into variable, kind and window.

    Input: model name -> feature column names, as made by sepsis_icu.features.build_features.
    Columns: model, feature, variable, kind, window_hours (NaN if none), is_missingness_indicator.
    Columns that do not match the feature naming (for example ICULOS) get kind "other".
    """
    rows = []
    for model, columns in columns_by_model.items():
        for column in columns:
            record = {
                "model": model, "feature": column, "variable": column, "kind": "other",
                "window_hours": float("nan"), "is_missingness_indicator": False,
            }
            for pattern, kind, windowed, missingness in _FEATURE_PATTERNS:
                match = pattern.match(column)
                if match:
                    record["variable"] = match.group("var")
                    record["is_missingness_indicator"] = missingness
                    if windowed:
                        record["kind"] = _STAT_NAMES[match.group("stat")]
                        record["window_hours"] = float(match.group("window"))
                    else:
                        record["kind"] = kind
                    break
            rows.append(record)
    return pd.DataFrame(rows)


# ---------- saving and loading models ----------

def _versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover
        pass
    return versions


def _model_paths(path: str) -> Tuple[Path, Path]:
    path = Path(path)
    model_path = path if path.suffix == ".joblib" else path.parent / (path.name + ".joblib")
    return model_path, model_path.with_suffix(".json")


def save_model(model, path: str, metadata: Optional[dict] = None) -> Tuple[Path, Path]:
    """Save a fitted model with joblib, plus a .json file next to it.

    The .json holds your metadata (for example the feature columns, the threshold, the training
    data), the package versions, and the save time. Returns (model_path, metadata_path).
    """
    import joblib

    model_path, meta_path = _model_paths(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    info = {"saved_utc": datetime.now(timezone.utc).isoformat(), "versions": _versions(), **(metadata or {})}
    meta_path.write_text(json.dumps(info, indent=2, default=str))
    return model_path, meta_path


def load_model(path: str):
    """Load a model saved by save_model. Returns (model, metadata).

    Warns if the scikit-learn version differs from the one that saved the model. Only load files
    you made yourself: joblib files are pickles.
    """
    import joblib

    model_path, meta_path = _model_paths(path)
    if not model_path.exists():
        raise FileNotFoundError(f"no model file at {model_path}")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    saved = metadata.get("versions", {}).get("scikit-learn")
    current = _versions().get("scikit-learn")
    if saved and current and saved != current:
        warnings.warn(
            f"this model was saved with scikit-learn {saved} but scikit-learn {current} is installed; "
            f"it may not load, or may predict differently",
            UserWarning,
            stacklevel=2,
        )
    return joblib.load(model_path), metadata


# ---------- saving and loading calibrators ----------

def calibrator_to_dict(calibrator) -> dict:
    """Turn a fitted calibrator into a plain dict (type plus its numbers)."""
    if isinstance(calibrator, PlattScaler):
        if calibrator.slope_ is None:
            raise ValueError("the calibrator has not been fit")
        return {"type": "platt", "intercept": float(calibrator.intercept_), "slope": float(calibrator.slope_)}
    if isinstance(calibrator, InterceptShift):
        if calibrator.shift_ is None:
            raise ValueError("the calibrator has not been fit")
        return {"type": "intercept_shift", "shift": float(calibrator.shift_)}
    if isinstance(calibrator, IsotonicCalibrator):
        if calibrator.x_ is None:
            raise ValueError("the calibrator has not been fit")
        return {"type": "isotonic", "x": calibrator.x_.tolist(), "y": calibrator.y_.tolist()}
    raise ValueError(f"unknown calibrator type: {type(calibrator).__name__}")


def calibrator_from_dict(data: dict):
    """Rebuild a calibrator from calibrator_to_dict output."""
    kind = data.get("type")
    if kind == "platt":
        calibrator = PlattScaler()
        calibrator.intercept_, calibrator.slope_ = float(data["intercept"]), float(data["slope"])
    elif kind == "intercept_shift":
        calibrator = InterceptShift()
        calibrator.shift_ = float(data["shift"])
    elif kind == "isotonic":
        calibrator = IsotonicCalibrator()
        calibrator.x_, calibrator.y_ = np.asarray(data["x"], dtype=float), np.asarray(data["y"], dtype=float)
    else:
        raise ValueError(f"unknown calibrator type: {kind!r}")
    return calibrator


def save_calibrator(calibrator, path: str, metadata: Optional[dict] = None) -> Path:
    """Write a fitted calibrator to a JSON file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**calibrator_to_dict(calibrator), "metadata": metadata or {}}, indent=2, default=str))
    return path


def load_calibrator(path: str):
    """Load a calibrator written by save_calibrator."""
    return calibrator_from_dict(json.loads(Path(path).read_text()))
