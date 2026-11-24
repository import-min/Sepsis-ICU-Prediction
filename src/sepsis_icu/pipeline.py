"""End-to-end run of the study (study_design.md, sections 3 to 10).

    python -m sepsis_icu.pipeline --config configs/default.yaml            # development only
    python -m sepsis_icu.pipeline --config configs/default.yaml --final    # + holdout and hospital B

Two modes, on purpose:

  Development (default)  Loads the data, builds targets and features, makes the patient-level
                         split, and cross-validates every model on the development patients.
                         Thresholds, lead-time bins, subgroups and error analysis are all computed
                         from those out-of-fold predictions. The patient-level holdout and
                         hospital B are NOT evaluated. Run this as often as you like.

  Final (--final)        Does everything above, then fits each model once on ALL development
                         patients and evaluates it on the holdout and on hospital B, using the
                         thresholds and recalibrators chosen on development data only. It writes
                         results/final_run.json and refuses to run again unless you pass
                         --force-repeat-final, so the holdout is touched once, at the end.

Fixed settings (seeds, windows, alert rate, bootstrap size, bins) come from the config file.
Nothing is written by hand: every number in results/ comes from this script.

Models and baselines compared (study_design.md, section 8):
  icu_los, sirs, qsofa, news2_style                     simple baselines
  logistic_regression_latest                            logistic regression on latest values
  gradient_boosting_with_missingness                    main model (rolling windows + missingness indicators)
  gradient_boosting_without_missingness                 same, without missingness indicators
All are judged at the same threshold rule: the alert rate set in the config
(alerting.target_alerts_per_patient_day), found on development data.

Output layout (results_dir from the config, default "results"):
  run_info.json                      config, versions, flags, warnings, files written
  data_summary.csv                   patients, prevalence, onset checks (left-censored, label turns off)
  splits/hospital_a_split.csv        which patient is in the holdout or in which fold
  dev/                               out-of-fold results on the development patients
  final/holdout/, final/hospital_b/  results on the holdout and hospital B (only with --final)
  final/shift_holdout_vs_hospital_b.csv
  manual_error_review.csv            sample of error cases to review by hand (path from config)
  report/                            analysis_counts.csv (patients and events in every analysis),
                                     characteristics.csv, characteristics_shift_a_vs_b.csv (final only),
                                     feature_inventory.csv (every feature each model used)
  final/artifacts/models/            each fitted model (.joblib) with a .json file of its feature
                                     columns, threshold and versions (load with reporting.load_model)
  final/artifacts/calibrators/       fitted calibrators as JSON (reporting.load_calibrator)
  final_run.json                     written once by --final

Each results folder holds: metrics.csv (with patient-level bootstrap CIs), thresholds, reliability.csv,
lead_time_bins.csv, detection.csv, utility.csv, trajectories.csv, subgroups.csv, false_alarms.csv,
false_alarm_profile.csv, miss_profile.csv, label_noise.csv; final folders also hold calibration_hour.csv
and calibration_patient.csv.

Not done here: hyperparameter search (models use fixed defaults from sepsis_icu.models). If you add
tuning, do it with the patient-level folds only.

Memory: rolling-window features create several hundred columns. They are built once per hospital and
stored as float32. Use --limit for a quick trial on the first N files per hospital.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from sepsis_icu.baselines import baseline_scores
from sepsis_icu.data import load_from_config, summarize, value_columns
from sepsis_icu.errors import (
    add_subgroups,
    build_review_table,
    false_alarm_profile,
    false_alarm_table,
    label_noise_candidates,
    miss_profile,
    patient_table,
    shift_comparison,
    subgroup_metrics,
    update_review_file,
)
from sepsis_icu.features import build_features
from sepsis_icu.leadtime import (
    bin_summary,
    detection_summary,
    match_controls,
    normalized_utility,
    patient_lead_times,
    score_trajectories,
)
from sepsis_icu.metrics import (
    bootstrap_ci,
    evaluate,
    operating_point,
    patient_level,
    reliability_table,
    threshold_for_alert_rate,
)
from sepsis_icu.models import (
    build_target,
    cross_validated_predictions,
    feature_columns,
    make_gradient_boosting,
    make_logistic_regression,
)
from sepsis_icu.recalibration import (
    InterceptShift,
    IsotonicCalibrator,
    PlattScaler,
    compare_calibration,
    split_labeled_portion,
)
from sepsis_icu.reporting import (
    analysis_counts,
    characteristics_shift,
    characteristics_table,
    feature_inventory,
    save_calibrator,
    save_model,
)
from sepsis_icu.splits import make_patient_split, rows_in, save_split

ID = "patient_id"
TIME = "ICULOS"
LABEL = "SepsisLabel"

BASELINE_COLUMNS = ("icu_los", "sirs", "qsofa", "news2_style")
SUBGROUP_VARIABLES = ("age_group", "sex", "unit", "stay_group", "missing_group")
PROFILE_COLUMNS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate", "Creatinine", "Platelets",
)
FINAL_MARKER = "final_run.json"


# ---------- settings ----------

def _opt(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass(frozen=True)
class Settings:
    """Everything the run depends on, read from the config file (with defaults)."""

    seed: int = 42
    horizon_hours: float = 6.0
    label_shift_hours: float = 6.0
    onset_verified: bool = False
    exclude_left_censored: bool = True
    n_splits: int = 5
    holdout_fraction: float = 0.2
    holdout_seed: int = 42
    windows: tuple = (3, 6, 12)
    missingness_variants: tuple = (True, False)
    target_alerts_per_patient_day: float = 1.0
    early_min_hours: float = 12.0
    intermediate_min_hours: float = 6.0
    trajectory_window_hours: float = 24.0
    n_resamples: int = 1000
    ci_level: float = 0.95
    ece_bins: int = 10
    labeled_fraction_b: float = 0.2
    results_dir: str = "results"
    review_path: str = "results/manual_error_review.csv"
    main_model: str = "gradient_boosting_with_missingness"

    @classmethod
    def from_config(cls, cfg: dict, n_resamples: Optional[int] = None, results_dir: Optional[str] = None) -> "Settings":
        d = cls()
        out_dir = results_dir or _opt(cfg, "paths", "results_dir", default=d.results_dir)
        review = _opt(cfg, "paths", "error_cases_table", default=None)
        if review is None or results_dir:
            review = str(Path(out_dir) / "manual_error_review.csv")
        return cls(
            seed=int(_opt(cfg, "seed", default=d.seed)),
            horizon_hours=float(_opt(cfg, "task", "prediction_horizon_hours", default=d.horizon_hours)),
            label_shift_hours=float(_opt(cfg, "labels", "label_shift_hours", default=d.label_shift_hours)),
            onset_verified=bool(_opt(cfg, "labels", "onset_verified", default=d.onset_verified)),
            exclude_left_censored=bool(_opt(cfg, "labels", "exclude_left_censored_from_lead_time", default=True)),
            n_splits=int(_opt(cfg, "splits", "n_splits", default=d.n_splits)),
            holdout_fraction=float(_opt(cfg, "splits", "holdout_fraction", default=d.holdout_fraction)),
            holdout_seed=int(_opt(cfg, "splits", "holdout_seed", default=d.holdout_seed)),
            windows=tuple(int(w) for w in _opt(cfg, "features", "rolling_windows_hours", default=list(d.windows))),
            missingness_variants=tuple(bool(v) for v in _opt(cfg, "features", "missingness_indicators", default=list(d.missingness_variants))),
            target_alerts_per_patient_day=float(_opt(cfg, "alerting", "target_alerts_per_patient_day", default=d.target_alerts_per_patient_day)),
            early_min_hours=float(_opt(cfg, "lead_time_bins", "early_min_hours", default=d.early_min_hours)),
            intermediate_min_hours=float(_opt(cfg, "lead_time_bins", "intermediate_min_hours", default=d.intermediate_min_hours)),
            trajectory_window_hours=float(_opt(cfg, "lead_time_bins", "trajectory_window_hours", default=d.trajectory_window_hours)),
            n_resamples=int(n_resamples if n_resamples is not None else _opt(cfg, "bootstrap", "n_resamples", default=d.n_resamples)),
            ci_level=float(_opt(cfg, "bootstrap", "ci_level", default=d.ci_level)),
            ece_bins=int(_opt(cfg, "calibration", "ece_bins", default=d.ece_bins)),
            labeled_fraction_b=float(_opt(cfg, "calibration", "external_intercept_update", "labeled_fraction_of_hospital_b", default=d.labeled_fraction_b)),
            results_dir=str(out_dir),
            review_path=str(review),
            main_model=str(_opt(cfg, "pipeline", "main_model", default=d.main_model)),
        )


def model_specs(s: Settings) -> Dict[str, tuple]:
    """name -> (feature kind, include missingness indicators, model factory)."""
    specs = {
        "logistic_regression_latest": ("latest", True, partial(make_logistic_regression, random_state=s.seed)),
    }
    if True in s.missingness_variants:
        specs["gradient_boosting_with_missingness"] = ("all", True, partial(make_gradient_boosting, random_state=s.seed))
    if False in s.missingness_variants:
        specs["gradient_boosting_without_missingness"] = ("all", False, partial(make_gradient_boosting, random_state=s.seed))
    return specs


# ---------- small helpers ----------

def _write(table: pd.DataFrame, path: Path, files: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    if str(path) not in files:
        files.append(str(path))


def _prepare(df: pd.DataFrame, vcols: List[str], s: Settings):
    """Target, features and baseline scores for one hospital, row-aligned with df."""
    y = build_target(df, s.horizon_hours, s.label_shift_hours).to_numpy()
    feats = build_features(df, vcols, windows=s.windows, missingness_indicators=True)
    base = baseline_scores(df)
    for other in (feats, base):
        same = (other[ID].to_numpy() == df[ID].to_numpy()).all() and (other[TIME].to_numpy() == df[TIME].to_numpy()).all()
        if not same:
            raise RuntimeError("features or baselines are not row-aligned with the data")
    columns = [c for c in feats.columns if c not in (ID, TIME)]
    feats = feats.astype({c: "float32" for c in columns})
    return y, feats, base


def _metrics_table(ids, y, scores, thresholds, prob, s: Settings) -> pd.DataFrame:
    tables = []
    for model, sc in scores.items():
        def fn(a, b, thr=thresholds[model], p=prob[model]):
            return evaluate(a, b, thr, p, s.ece_bins)

        if s.n_resamples > 0:
            t = bootstrap_ci(ids, y, sc, fn, n_resamples=s.n_resamples, ci_level=s.ci_level, seed=s.seed)
        else:
            est = fn(y, sc)
            t = pd.DataFrame(
                {"metric": list(est), "estimate": list(est.values()), "ci_lower": np.nan, "ci_upper": np.nan, "n_valid": 0}
            )
        t.insert(0, "model", model)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


# ---------- analysis of one set of patients ----------

def analyze_slice(
    outdir: Path,
    df: pd.DataFrame,
    y: np.ndarray,
    scores: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    prob: Dict[str, bool],
    s: Settings,
    files: List[str],
    vcols: List[str],
    log: Callable[[str], None],
) -> Optional[pd.DataFrame]:
    """Every table for one set of patients and their scores. Returns the manual-review sample
    for the main model."""
    ids = df[ID].to_numpy()
    hours = df[TIME].to_numpy()
    label = df[LABEL].to_numpy()

    log(f"  metrics with bootstrap intervals ({s.n_resamples} resamples)")
    _write(_metrics_table(ids, y, scores, thresholds, prob, s), outdir / "metrics.csv", files)

    reliability = []
    for model, sc in scores.items():
        if prob[model]:
            t = reliability_table(y, sc, s.ece_bins)
            t.insert(0, "model", model)
            reliability.append(t)
    if reliability:
        _write(pd.concat(reliability, ignore_index=True), outdir / "reliability.csv", files)

    log("  lead-time bins, detection, utility, trajectories")
    bins, detection, utility, trajectories = [], [], [], []
    lead_by_model: Dict[str, pd.DataFrame] = {}
    pairs = None
    for model, sc in scores.items():
        frame = pd.DataFrame({ID: ids, TIME: hours, LABEL: label, "score": sc})
        lead = patient_lead_times(
            frame, "score", thresholds[model], s.label_shift_hours, s.early_min_hours, s.intermediate_min_hours
        )
        lead_by_model[model] = lead
        b = bin_summary(lead, s.exclude_left_censored, s.n_resamples, s.ci_level, s.seed)
        b.insert(0, "model", model)
        bins.append(b)
        d = detection_summary(lead, s.exclude_left_censored)
        d["model"] = model
        detection.append(d)
        u = normalized_utility(frame, "score", thresholds[model], s.label_shift_hours)
        u["model"] = model
        utility.append(u)
        if pairs is None:  # matching depends only on onset and stay length, not on the model
            usable = lead[~lead["left_censored"]] if s.exclude_left_censored else lead
            pairs = match_controls(frame, usable, s.seed)
        tr = score_trajectories(frame, "score", pairs, s.trajectory_window_hours)
        tr.insert(0, "model", model)
        trajectories.append(tr)
    _write(pd.concat(bins, ignore_index=True), outdir / "lead_time_bins.csv", files)
    _write(pd.DataFrame(detection), outdir / "detection.csv", files)
    _write(pd.DataFrame(utility), outdir / "utility.csv", files)
    _write(pd.concat(trajectories, ignore_index=True), outdir / "trajectories.csv", files)

    log(f"  error analysis for {s.main_model}")
    main = s.main_model
    lead_main = lead_by_model[main]
    with_score = df.assign(score=scores[main])
    patients = add_subgroups(patient_table(df, vcols))

    eval_frame = pd.DataFrame({ID: ids, "y": y, "score": scores[main]})
    subgroups = [
        subgroup_metrics(eval_frame, patients, var, "score", thresholds[main], prob[main], lead_table=lead_main, label_col="y")
        for var in SUBGROUP_VARIABLES
    ]
    _write(pd.concat(subgroups, ignore_index=True), outdir / "subgroups.csv", files)
    _write(false_alarm_table(with_score, patients, "score", thresholds[main], by="unit"), outdir / "false_alarms.csv", files)

    profile_cols = [c for c in PROFILE_COLUMNS if c in df.columns]
    if profile_cols:
        _write(false_alarm_profile(with_score, patients, "score", profile_cols, observed_fraction=True), outdir / "false_alarm_profile.csv", files)
        _write(miss_profile(with_score, lead_main, profile_cols, observed_fraction=True), outdir / "miss_profile.csv", files)
    _write(label_noise_candidates(with_score, "score", thresholds[main], lead_main), outdir / "label_noise.csv", files)

    return build_review_table(with_score, "score", thresholds[main], lead_main, per_type=10, seed=s.seed)


# ---------- recalibration (final mode) ----------

def _calibration_table(
    p_dev, y_dev, p_target, y_target, labeled, s: Settings, warnings: List[str], tag: str, fitted: Optional[dict] = None
) -> pd.DataFrame:
    """Compare uncalibrated and recalibrated predictions on the target. `labeled` is
    (p, y) of the labeled portion of a new site, or None. Calibrators that could be fit are
    added to `fitted` (variant name -> calibrator) if it is given."""
    variants = {"uncalibrated": p_target}
    for label, calibrator in (("platt_dev", PlattScaler()), ("isotonic_dev", IsotonicCalibrator())):
        try:
            variants[label] = calibrator.fit(p_dev, y_dev).transform(p_target)
            if fitted is not None:
                fitted[label] = calibrator
        except ValueError as err:
            warnings.append(f"{tag}: {label} skipped ({err})")
    if labeled is not None:
        try:
            shift = InterceptShift().fit(*labeled)
            variants["intercept_update_labeled_b"] = shift.transform(p_target)
            if fitted is not None:
                fitted["intercept_update_labeled_b"] = shift
        except ValueError as err:
            warnings.append(f"{tag}: intercept update skipped ({err})")
    return compare_calibration(y_target, variants, s.ece_bins)


def calibration_reports(
    model: str,
    dev: tuple,
    target: tuple,
    labeled_ids: Optional[np.ndarray],
    s: Settings,
    warnings: List[str],
    dataset: str,
) -> Dict[str, pd.DataFrame]:
    """Hour-level and patient-level calibration before and after recalibration.

    dev and target are (patient_ids, y, p). Calibrators are fit on dev only. For a new site
    (labeled_ids given), the intercept update is fit on the labeled patients and EVERY variant
    is evaluated on the remaining patients.
    """
    ids_d, y_d, p_d = dev
    ids_t, y_t, p_t = target

    labeled_rows = None
    eval_rows = np.ones(len(ids_t), dtype=bool)
    if labeled_ids is not None:
        in_labeled = np.isin(ids_t, labeled_ids)
        labeled_rows = (p_t[in_labeled], y_t[in_labeled])
        eval_rows = ~in_labeled
    fitted_hour: Dict[str, object] = {}
    hour = _calibration_table(p_d, y_d, p_t[eval_rows], y_t[eval_rows], labeled_rows, s, warnings, f"{dataset}/{model}/hour", fitted_hour)

    ever_d, high_d = patient_level(ids_d, y_d, p_d)
    ever_t, high_t = patient_level(ids_t, y_t, p_t)
    unique_t = np.unique(ids_t)
    labeled_pat = None
    eval_pat = np.ones(len(unique_t), dtype=bool)
    if labeled_ids is not None:
        in_labeled = np.isin(unique_t, labeled_ids)
        labeled_pat = (high_t[in_labeled], ever_t[in_labeled])
        eval_pat = ~in_labeled
    fitted_patient: Dict[str, object] = {}
    patient = _calibration_table(high_d, ever_d, high_t[eval_pat], ever_t[eval_pat], labeled_pat, s, warnings, f"{dataset}/{model}/patient", fitted_patient)

    hour.insert(0, "model", model)
    patient.insert(0, "model", model)
    return {"hour": hour, "patient": patient, "calibrators": {"hour": fitted_hour, "patient": fitted_patient}}


# ---------- the run ----------

def run(
    config_path: str,
    final: bool = False,
    force_repeat_final: bool = False,
    limit: Optional[int] = None,
    n_resamples: Optional[int] = None,
    results_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the study. Returns {"results_dir", "files", "warnings", "final"}."""
    started = datetime.now(timezone.utc)
    cfg = yaml.safe_load(Path(config_path).read_text())
    s = Settings.from_config(cfg, n_resamples=n_resamples, results_dir=results_dir)
    out = Path(s.results_dir)

    marker = out / FINAL_MARKER
    if final and marker.exists() and not force_repeat_final:
        done = json.loads(marker.read_text()).get("finished_utc", "an earlier run")
        raise RuntimeError(
            f"The holdout and hospital B were already evaluated ({done}). The study design says they are "
            f"touched once. Pass force_repeat_final / --force-repeat-final to run again, and log why in "
            f"study_design.md."
        )

    specs = model_specs(s)
    if s.main_model not in specs:
        raise ValueError(f"main_model '{s.main_model}' is not one of {sorted(specs)}")

    files: List[str] = []
    warnings: List[str] = []
    if not s.onset_verified:
        warnings.append(
            "labels.onset_verified is false: the 6-hour label shift has not been checked against the "
            "Challenge documentation. Every lead-time result depends on it."
        )
        log("WARNING: " + warnings[-1])

    log("loading data")
    dev_df, ext_df, _ = load_from_config(config_path, limit)  # also proves A and B share no patients
    vcols = value_columns(dev_df)
    summaries = [{"hospital": "A", **summarize(dev_df, s.label_shift_hours)}]

    log("building targets, features and baseline scores for hospital A")
    y_a, feats_a, base_a = _prepare(dev_df, vcols, s)
    pids = dev_df[ID].to_numpy()

    log("making the patient-level split")
    split = make_patient_split(dev_df[ID], dev_df[LABEL], s.holdout_fraction, s.n_splits, s.holdout_seed, s.seed)
    save_split(str(out / "splits" / "hospital_a_split.csv"), split)
    files.append(str(out / "splits" / "hospital_a_split.csv"))
    holdout_mask = rows_in(pids, split.holdout_ids())
    dev_mask = ~holdout_mask

    _write(analysis_counts(dev_df, y_a, split), out / "report" / "analysis_counts.csv", files)
    patients_dev = patient_table(dev_df.loc[dev_mask], vcols)
    _write(characteristics_table({"hospital_a_development": patients_dev}), out / "report" / "characteristics.csv", files)

    prob: Dict[str, bool] = {name: True for name in specs}
    prob.update({name: False for name in BASELINE_COLUMNS})

    log(f"cross-validating {len(specs)} models on the development patients ({s.n_splits} folds)")
    columns_for = {name: feature_columns(feats_a, kind=kind, include_missingness=inc) for name, (kind, inc, _) in specs.items()}
    _write(feature_inventory(columns_for), out / "report" / "feature_inventory.csv", files)
    oof: Dict[str, np.ndarray] = {}
    for name, (kind, inc, factory) in specs.items():
        log(f"  {name}")
        X = feats_a[columns_for[name]].to_numpy(dtype=np.float32)
        oof[name] = cross_validated_predictions(X, y_a, pids, split, factory)

    dev_scores = {name: oof[name][dev_mask] for name in specs}
    dev_scores.update({name: base_a[name].to_numpy(dtype=float)[dev_mask] for name in BASELINE_COLUMNS})
    dev_y = y_a[dev_mask]
    dev_rows = dev_df.loc[dev_mask].reset_index(drop=True)

    thresholds = {
        name: threshold_for_alert_rate(sc, s.target_alerts_per_patient_day) for name, sc in dev_scores.items()
    }
    threshold_table = pd.DataFrame(
        {
            "model": list(thresholds),
            "threshold": list(thresholds.values()),
            "target_alerts_per_patient_day": s.target_alerts_per_patient_day,
            "development_alerts_per_patient_day": [
                operating_point(dev_y, dev_scores[m], thresholds[m])["alerts_per_patient_day"] for m in thresholds
            ],
        }
    )
    _write(threshold_table, out / "dev" / "thresholds.csv", files)

    log("analysing development (out-of-fold) predictions")
    analyze_slice(out / "dev", dev_rows, dev_y, dev_scores, thresholds, prob, s, files, vcols, log)

    if final:
        log("FINAL: fitting each model once on all development patients")
        y_b, feats_b, base_b = _prepare(ext_df, vcols, s)
        summaries.append({"hospital": "B", **summarize(ext_df, s.label_shift_hours)})

        hold_scores: Dict[str, np.ndarray] = {}
        b_scores: Dict[str, np.ndarray] = {}
        for name, (kind, inc, factory) in specs.items():
            log(f"  {name}")
            X_a = feats_a[columns_for[name]].to_numpy(dtype=np.float32)
            X_b = feats_b[columns_for[name]].to_numpy(dtype=np.float32)
            model = factory().fit(X_a[dev_mask], y_a[dev_mask])
            model_path, meta_path = save_model(
                model,
                str(out / "final" / "artifacts" / "models" / name),
                metadata={
                    "model": name,
                    "feature_columns": columns_for[name],
                    "feature_kind": kind,
                    "missingness_indicators": inc,
                    "threshold": thresholds[name],
                    "target_alerts_per_patient_day": s.target_alerts_per_patient_day,
                    "horizon_hours": s.horizon_hours,
                    "label_shift_hours": s.label_shift_hours,
                    "rolling_windows_hours": list(s.windows),
                    "seed": s.seed,
                    "trained_on": {
                        "dataset": "hospital_a_development",
                        "patients": int(len(split.development_ids())),
                        "rows": int(dev_mask.sum()),
                    },
                },
            )
            files.extend([str(model_path), str(meta_path)])
            hold_scores[name] = model.predict_proba(X_a[holdout_mask])[:, 1]
            b_scores[name] = model.predict_proba(X_b)[:, 1]
        for name in BASELINE_COLUMNS:
            hold_scores[name] = base_a[name].to_numpy(dtype=float)[holdout_mask]
            b_scores[name] = base_b[name].to_numpy(dtype=float)

        hold_df = dev_df.loc[holdout_mask].reset_index(drop=True)
        hold_y = y_a[holdout_mask]
        patients_holdout = patient_table(hold_df, vcols)
        patients_b = patient_table(ext_df, vcols)
        _write(
            characteristics_table(
                {"hospital_a_development": patients_dev, "hospital_a_holdout": patients_holdout, "hospital_b": patients_b}
            ),
            out / "report" / "characteristics.csv", files,
        )
        _write(
            characteristics_shift(patient_table(dev_df, vcols), patients_b),
            out / "report" / "characteristics_shift_a_vs_b.csv", files,
        )
        _write(analysis_counts(dev_df, y_a, split, external=(ext_df, y_b)), out / "report" / "analysis_counts.csv", files)
        reviews = []
        for tag, sub_df, sub_y, sub_scores in (
            ("holdout", hold_df, hold_y, hold_scores),
            ("hospital_b", ext_df, y_b, b_scores),
        ):
            log(f"analysing {tag}")
            folder = out / "final" / tag
            _write(threshold_table, folder / "thresholds.csv", files)
            reviews.append(analyze_slice(folder, sub_df, sub_y, sub_scores, thresholds, prob, s, files, vcols, log))

        log("recalibration (fit on development predictions only; intercept update on labeled part of B)")
        labeled_ids, _ = split_labeled_portion(ext_df[ID], ext_df[LABEL], s.labeled_fraction_b, s.seed)
        dev_ids = dev_rows[ID].to_numpy()
        for tag, sub_df, sub_y, sub_scores, labeled in (
            ("holdout", hold_df, hold_y, hold_scores, None),
            ("hospital_b", ext_df, y_b, b_scores, labeled_ids),
        ):
            hour_tables, patient_tables = [], []
            for name in specs:
                rep = calibration_reports(
                    name,
                    (dev_ids, dev_y, dev_scores[name]),
                    (sub_df[ID].to_numpy(), sub_y, sub_scores[name]),
                    labeled, s, warnings, tag,
                )
                hour_tables.append(rep["hour"])
                patient_tables.append(rep["patient"])
                # The development-fit calibrators are the same for both datasets, so save them once
                # (from the holdout call). The intercept update only exists for hospital B.
                for level, fitted in rep["calibrators"].items():
                    for variant, calibrator in fitted.items():
                        if (tag == "holdout") == (variant != "intercept_update_labeled_b"):
                            path = save_calibrator(
                                calibrator,
                                str(out / "final" / "artifacts" / "calibrators" / f"{name}_{level}_{variant}.json"),
                                metadata={
                                    "model": name, "level": level, "variant": variant,
                                    "fit_on": "labeled part of hospital_b" if variant == "intercept_update_labeled_b"
                                    else "development out-of-fold predictions",
                                },
                            )
                            files.append(str(path))
            _write(pd.concat(hour_tables, ignore_index=True), out / "final" / tag / "calibration_hour.csv", files)
            _write(pd.concat(patient_tables, ignore_index=True), out / "final" / tag / "calibration_patient.csv", files)

        log("shift between the holdout and hospital B")
        shifts = []
        for name in dev_scores:
            t = shift_comparison(hold_y, hold_scores[name], y_b, b_scores[name], thresholds[name], prob[name])
            t.insert(0, "model", name)
            shifts.append(t)
        _write(pd.concat(shifts, ignore_index=True), out / "final" / "shift_holdout_vs_hospital_b.csv", files)

        review = pd.concat([r for r in reviews if r is not None], ignore_index=True)
        update_review_file(s.review_path, review)
        files.append(s.review_path)

    _write(pd.DataFrame(summaries), out / "data_summary.csv", files)

    finished = datetime.now(timezone.utc)
    info = {
        "config_path": str(config_path),
        "config": cfg,
        "settings": asdict(s),
        "final": final,
        "limit": limit,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "python": platform.python_version(),
        "packages": {name: sys.modules[name].__version__ for name in ("numpy", "pandas", "sklearn") if name in sys.modules},
        "n_patients_hospital_a": int(dev_df[ID].nunique()),
        "n_patients_holdout": int(len(split.holdout_ids())),
        "n_patients_hospital_b_loaded": int(ext_df[ID].nunique()),
        "warnings": warnings,
        "files": files,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    if final:
        marker.write_text(json.dumps({"finished_utc": finished.isoformat(), "config_path": str(config_path)}, indent=2))
    log(f"done: {len(files)} files in {out}")
    return {"results_dir": str(out), "files": files, "warnings": warnings, "final": final}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the sepsis early-warning study end to end.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--final", action="store_true", help="also evaluate the holdout and hospital B (once)")
    parser.add_argument("--force-repeat-final", action="store_true", help="run --final again; log why in study_design.md")
    parser.add_argument("--limit", type=int, default=None, help="only load the first N files per hospital (quick trial)")
    parser.add_argument("--n-resamples", type=int, default=None, help="override the bootstrap size in the config")
    parser.add_argument("--out", default=None, help="override the results folder")
    parser.add_argument("--require-verified-onset", action="store_true", help="stop if labels.onset_verified is false")
    args = parser.parse_args(argv)

    if args.require_verified_onset:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if not _opt(cfg, "labels", "onset_verified", default=False):
            raise SystemExit("labels.onset_verified is false. Check the 6-hour label shift against the Challenge documentation first.")

    run(
        args.config,
        final=args.final,
        force_repeat_final=args.force_repeat_final,
        limit=args.limit,
        n_resamples=args.n_resamples,
        results_dir=args.out,
    )


if __name__ == "__main__":
    main()
