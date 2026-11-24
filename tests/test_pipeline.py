"""End-to-end tests for the pipeline, on small fake Challenge-style data.

The most important tests change hospital B, or the holdout patients' values, and check that
results that must not depend on them do not change at all.

Skipped automatically if scikit-learn is not installed.

Run from the repo root:  pytest -q
"""

import json
import shutil
import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")

from sepsis_icu.errors import REVIEW_COLUMNS
from sepsis_icu.reporting import load_calibrator, load_model
from sepsis_icu.pipeline import (
    BASELINE_COLUMNS,
    Settings,
    calibration_reports,
    main,
    model_specs,
    run,
)

MODELS = {
    "logistic_regression_latest",
    "gradient_boosting_with_missingness",
    "gradient_boosting_without_missingness",
}
ALL_MODELS = MODELS | set(BASELINE_COLUMNS)

DEV_FILES = [
    "metrics.csv", "thresholds.csv", "reliability.csv", "lead_time_bins.csv", "detection.csv",
    "utility.csv", "trajectories.csv", "subgroups.csv", "false_alarms.csv", "false_alarm_profile.csv",
    "miss_profile.csv", "label_noise.csv",
]


# ---------- fake data ----------

def make_patient(rng, n, septic, hr_shift=0.0):
    hours = np.arange(1, n + 1)
    label = np.zeros(n, dtype=int)
    severity = np.zeros(n)
    if septic:
        start = int(rng.integers(5, n - 2))              # label turns on at this hour
        label[start - 1:] = 1
        severity = np.clip((hours - (start - 6)) / 6.0, 0, 1)

    def measure(mean, slope, sd, p_missing):
        values = rng.normal(mean + slope * severity, sd, n)
        values[rng.random(n) < p_missing] = np.nan
        return values

    return pd.DataFrame(
        {
            "HR": measure(80 + hr_shift, 25, 8, 0.3),
            "O2Sat": measure(97, -4, 1.5, 0.3),
            "Temp": measure(36.9, 1.3, 0.4, 0.5),
            "SBP": measure(120, -25, 10, 0.3),
            "MAP": measure(85, -15, 7, 0.4),
            "DBP": measure(65, -10, 6, 0.4),
            "Resp": measure(16, 8, 2.5, 0.3),
            "WBC": measure(8, 6, 2, 0.8),
            "PaCO2": measure(40, -8, 3, 0.8),
            "Lactate": measure(1.2, 2.5, 0.5, 0.85),
            "Age": float(rng.integers(25, 90)),
            "Gender": int(rng.integers(0, 2)),
            "Unit1": int(rng.integers(0, 2)),
            "Unit2": int(rng.integers(0, 2)),
            "HospAdmTime": float(rng.normal(-3, 2)),
            "ICULOS": hours,
            "SepsisLabel": label,
        }
    )


def write_hospital(directory, n_patients, seed, hr_shift=0.0):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_patients):
        n = int(rng.integers(12, 28))
        patient = make_patient(rng, n, septic=bool(rng.random() < 0.3), hr_shift=hr_shift)
        patient.to_csv(directory / f"p{i + 1:06d}.psv", sep="|", index=False, na_rep="NaN")


def config_dict(a_dir, b_dir, **overrides):
    cfg = {
        "seed": 7,
        "task": {"prediction_horizon_hours": 6},
        "labels": {"label_shift_hours": 6, "onset_verified": False, "exclude_left_censored_from_lead_time": True},
        "data": {
            "development_dir": str(a_dir),
            "external_test_dir": str(b_dir),
            "patient_id_column": "patient_id",
            "time_column": "ICULOS",
            "label_column": "SepsisLabel",
        },
        "splits": {"n_splits": 3, "holdout_fraction": 0.25, "holdout_seed": 11},
        "features": {"rolling_windows_hours": [3, 6], "missingness_indicators": [True, False]},
        "alerting": {"target_alerts_per_patient_day": 2.0},
        "lead_time_bins": {"early_min_hours": 12, "intermediate_min_hours": 6, "trajectory_window_hours": 24},
        "bootstrap": {"n_resamples": 15, "ci_level": 0.95},
        "calibration": {"ece_bins": 10, "external_intercept_update": {"labeled_fraction_of_hospital_b": 0.3}},
    }
    cfg.update(overrides)
    return cfg


def make_world(root, seed_a=1, seed_b=2, n_a=120, n_b=70, **config_overrides):
    """Write hospital A, hospital B and a config file. Returns the config path."""
    root.mkdir(parents=True, exist_ok=True)
    write_hospital(root / "A", n_a, seed_a)
    write_hospital(root / "B", n_b, seed_b, hr_shift=3.0)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict(root / "A", root / "B", **config_overrides)))
    return str(path)


def quiet(_message):
    pass


def text(path):
    return Path(path).read_text()


# ---------- shared runs ----------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="module")
def dev_run(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("dev_run")
    result = run(world, results_dir=str(out), log=quiet)
    return Path(result["results_dir"]), result


@pytest.fixture(scope="module")
def final_run(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("final_run")
    result = run(world, final=True, results_dir=str(out), log=quiet)
    return Path(result["results_dir"]), result


# ---------- development mode ----------

def test_development_run_writes_every_table_and_nothing_final(dev_run):
    out, result = dev_run
    for name in DEV_FILES:
        assert (out / "dev" / name).exists(), name
    assert (out / "splits" / "hospital_a_split.csv").exists()
    assert (out / "data_summary.csv").exists() and (out / "run_info.json").exists()
    # The holdout and hospital B are not evaluated, and the final marker is not written.
    assert not (out / "final").exists()
    assert not (out / "final_run.json").exists()
    assert not (out / "manual_error_review.csv").exists()
    assert result["final"] is False


def test_development_run_does_not_summarize_hospital_b(dev_run):
    out, _ = dev_run
    summary = pd.read_csv(out / "data_summary.csv")
    assert summary["hospital"].tolist() == ["A"]


def test_metrics_cover_every_model_with_confidence_intervals(dev_run):
    out, _ = dev_run
    m = pd.read_csv(out / "dev" / "metrics.csv")
    assert set(m["model"]) == ALL_MODELS
    auroc = m[m["metric"] == "auroc"].set_index("model")
    assert auroc.loc["gradient_boosting_with_missingness", "estimate"] > 0.75   # the fake data has a real signal
    assert (auroc["ci_lower"] <= auroc["estimate"]).all() and (auroc["estimate"] <= auroc["ci_upper"]).all()
    assert (auroc["n_valid"] > 0).all()
    # Baseline scores are not probabilities, so they get no calibration metrics.
    brier = m[m["metric"] == "brier"].set_index("model")["estimate"]
    assert brier[list(BASELINE_COLUMNS)].isna().all() and brier[list(MODELS)].notna().all()


def test_thresholds_respect_the_alert_rate_set_in_the_config(dev_run):
    out, _ = dev_run
    t = pd.read_csv(out / "dev" / "thresholds.csv")
    assert set(t["model"]) == ALL_MODELS
    assert (t["target_alerts_per_patient_day"] == 2.0).all()
    assert (t["development_alerts_per_patient_day"] <= 2.0 + 1e-9).all()


def test_the_patient_split_follows_the_config(dev_run):
    out, _ = dev_run
    split = pd.read_csv(out / "splits" / "hospital_a_split.csv")
    assert len(split) == 120
    holdout = split[split["role"] == "holdout"]
    assert 0.2 <= len(holdout) / len(split) <= 0.3
    assert (holdout["fold"] == -1).all()
    assert set(split.loc[split["role"] == "development", "fold"]) == {0, 1, 2}


def test_run_info_records_the_onset_warning_and_the_files(dev_run):
    out, result = dev_run
    info = json.loads(text(out / "run_info.json"))
    assert info["final"] is False and info["settings"]["seed"] == 7
    assert any("onset_verified" in w for w in info["warnings"])
    assert any(f.endswith("metrics.csv") for f in info["files"])
    assert set(info["packages"]) >= {"numpy", "pandas"}
    assert result["warnings"] == info["warnings"]


def test_lead_time_and_error_tables_have_the_expected_models(dev_run):
    out, _ = dev_run
    bins = pd.read_csv(out / "dev" / "lead_time_bins.csv")
    assert set(bins["model"]) == ALL_MODELS and len(bins) == 4 * len(ALL_MODELS)
    util = pd.read_csv(out / "dev" / "utility.csv")
    assert set(util["model"]) == ALL_MODELS
    sub = pd.read_csv(out / "dev" / "subgroups.csv")
    assert set(sub["variable"]) == {"age_group", "sex", "unit", "stay_group", "missing_group"}
    assert (out / "dev" / "false_alarms.csv").stat().st_size > 0


def test_a_development_run_is_reproducible(world, dev_run, tmp_path):
    first, _ = dev_run
    again = run(world, results_dir=str(tmp_path / "again"), log=quiet)
    second = Path(again["results_dir"])
    for name in ("metrics.csv", "thresholds.csv", "lead_time_bins.csv", "subgroups.csv"):
        assert text(first / "dev" / name) == text(second / "dev" / name), name
    assert text(first / "splits" / "hospital_a_split.csv") == text(second / "splits" / "hospital_a_split.csv")


# ---------- leakage: development results must not depend on hospital B or the holdout ----------

def test_changing_hospital_b_does_not_change_development_results(world, dev_run, tmp_path):
    first, _ = dev_run
    other_world = make_world(tmp_path / "other_b", seed_a=1, seed_b=999)   # same A, a completely different B
    other = Path(run(other_world, results_dir=str(tmp_path / "out"), log=quiet)["results_dir"])
    for name in ("metrics.csv", "thresholds.csv", "lead_time_bins.csv", "utility.csv"):
        assert text(first / "dev" / name) == text(other / "dev" / name), name
    for name in ("analysis_counts.csv", "characteristics.csv", "feature_inventory.csv"):
        assert text(first / "report" / name) == text(other / "report" / name), name


def test_changing_the_values_of_holdout_patients_does_not_change_development_results(world, dev_run, tmp_path):
    first, _ = dev_run
    split = pd.read_csv(first / "splits" / "hospital_a_split.csv")
    holdout_files = [p.replace("A_", "", 1) + ".psv" for p in split.loc[split["role"] == "holdout", "patient_id"]]

    changed = tmp_path / "changed"
    shutil.copytree(Path(yaml.safe_load(text(world))["data"]["development_dir"]), changed / "A")
    shutil.copytree(Path(yaml.safe_load(text(world))["data"]["external_test_dir"]), changed / "B")
    for name in holdout_files:        # change measurements only (not labels, so the split is unchanged)
        path = changed / "A" / name
        data = pd.read_csv(path, sep="|")
        for col in ("HR", "Temp", "Resp", "SBP", "O2Sat"):
            data[col] = data[col] * 3 + 11
        data.to_csv(path, sep="|", index=False, na_rep="NaN")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(config_dict(changed / "A", changed / "B")))

    other = Path(run(str(config), results_dir=str(tmp_path / "out"), log=quiet)["results_dir"])
    for name in ("metrics.csv", "thresholds.csv", "lead_time_bins.csv"):
        assert text(first / "dev" / name) == text(other / "dev" / name), name


# ---------- final mode ----------

def test_final_run_writes_holdout_and_hospital_b_results(final_run):
    out, result = final_run
    for tag in ("holdout", "hospital_b"):
        for name in DEV_FILES[:1] + DEV_FILES[2:] + ["calibration_hour.csv", "calibration_patient.csv", "thresholds.csv"]:
            assert (out / "final" / tag / name).exists(), (tag, name)
    assert (out / "final" / "shift_holdout_vs_hospital_b.csv").exists()
    assert (out / "final_run.json").exists() and result["final"] is True
    summary = pd.read_csv(out / "data_summary.csv")
    assert summary["hospital"].tolist() == ["A", "B"]


def test_final_results_use_the_development_thresholds(dev_run, final_run):
    dev_out, _ = dev_run
    out, _ = final_run
    dev_thresholds = text(dev_out / "dev" / "thresholds.csv")
    assert text(out / "dev" / "thresholds.csv") == dev_thresholds
    for tag in ("holdout", "hospital_b"):
        assert text(out / "final" / tag / "thresholds.csv") == dev_thresholds


def test_final_metrics_cover_every_model_on_both_datasets(final_run):
    out, _ = final_run
    for tag in ("holdout", "hospital_b"):
        m = pd.read_csv(out / "final" / tag / "metrics.csv")
        assert set(m["model"]) == ALL_MODELS
        prevalence = m[(m["metric"] == "prevalence") & (m["model"] == "sirs")]["estimate"].iloc[0]
        assert 0 < prevalence < 1


def test_shift_table_compares_holdout_and_hospital_b(final_run):
    out, _ = final_run
    shift = pd.read_csv(out / "final" / "shift_holdout_vs_hospital_b.csv")
    assert set(shift["model"]) == ALL_MODELS
    assert list(shift.columns) == ["model", "metric", "hospital_a", "hospital_b", "difference"]


def test_calibration_tables_before_and_after_recalibration(final_run):
    out, _ = final_run
    hour_b = pd.read_csv(out / "final" / "hospital_b" / "calibration_hour.csv")
    assert set(hour_b["model"]) == MODELS
    assert {"uncalibrated", "platt_dev", "intercept_update_labeled_b"} <= set(hour_b["variant"])
    hour_holdout = pd.read_csv(out / "final" / "holdout" / "calibration_hour.csv")
    assert "intercept_update_labeled_b" not in set(hour_holdout["variant"])   # only a new site gets that
    patient_b = pd.read_csv(out / "final" / "hospital_b" / "calibration_patient.csv")
    assert "uncalibrated" in set(patient_b["variant"])


def test_the_manual_review_table_is_written_once_with_blank_columns(final_run):
    out, _ = final_run
    review = pd.read_csv(out / "manual_error_review.csv", dtype=str, keep_default_na=False)
    assert list(review.columns) == REVIEW_COLUMNS
    assert {"false_alarm", "never_alerted"} & set(review["error_type"])
    assert (review["finding"] == "").all()
    assert not review.duplicated(["patient_id", "error_type"]).any()


def test_final_mode_cannot_be_repeated_without_saying_so(world, final_run, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(final_run[0], out)          # work on a copy so the shared results stay untouched
    with pytest.raises(RuntimeError, match="touched once"):
        run(world, final=True, results_dir=str(out), log=quiet)
    # A development run in the same folder is still allowed.
    run(world, results_dir=str(out), n_resamples=0, log=quiet)


def test_the_repeat_guard_runs_before_any_data_is_loaded(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "final_run.json").write_text(json.dumps({"finished_utc": "2026-01-01T00:00:00+00:00"}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config_dict(tmp_path / "missing_A", tmp_path / "missing_B")))
    with pytest.raises(RuntimeError, match="2026-01-01"):
        run(str(cfg), final=True, results_dir=str(out), log=quiet)


def test_repeating_final_works_when_forced_and_without_bootstrap(world, final_run, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(final_run[0], out)          # work on a copy so the shared results stay untouched
    result = run(world, final=True, force_repeat_final=True, n_resamples=0, results_dir=str(out), log=quiet)
    assert result["final"] is True
    m = pd.read_csv(out / "final" / "holdout" / "metrics.csv")
    assert m["ci_lower"].isna().all()   # no bootstrap was asked for


def test_final_holdout_results_do_not_depend_on_hospital_b(final_run, tmp_path):
    out, _ = final_run
    other_world = make_world(tmp_path / "other_b", seed_a=1, seed_b=555)    # same A, different B
    other = Path(run(other_world, final=True, results_dir=str(tmp_path / "out"), log=quiet)["results_dir"])
    # Everything fit or chosen on development data gives the same holdout results.
    for name in ("metrics.csv", "calibration_hour.csv", "calibration_patient.csv", "lead_time_bins.csv"):
        assert text(out / "final" / "holdout" / name) == text(other / "final" / "holdout" / name), name
    # ... while hospital B's own results do change.
    assert text(out / "final" / "hospital_b" / "metrics.csv") != text(other / "final" / "hospital_b" / "metrics.csv")


# ---------- report tables and saved files ----------

def test_development_run_writes_report_tables_without_touching_hospital_b(dev_run):
    out, _ = dev_run
    counts = pd.read_csv(out / "report" / "analysis_counts.csv").set_index("analysis")
    assert counts.loc["hospital_a_all", "patients"] == 120
    assert counts.loc["hospital_a_development", "patients"] + counts.loc["hospital_a_holdout", "patients"] == 120
    assert {"fold_0_training", "fold_2_validation"} <= set(counts.index)
    assert "hospital_b" not in counts.index

    characteristics = pd.read_csv(out / "report" / "characteristics.csv")
    assert set(characteristics["dataset"]) == {"hospital_a_development"}
    assert not (out / "report" / "characteristics_shift_a_vs_b.csv").exists()

    inventory = pd.read_csv(out / "report" / "feature_inventory.csv")
    assert set(inventory["model"]) == MODELS
    flagged = inventory.groupby("model")["is_missingness_indicator"].any()
    assert flagged["gradient_boosting_with_missingness"] and not flagged["gradient_boosting_without_missingness"]
    assert not flagged["logistic_regression_latest"]
    assert (inventory.loc[inventory["model"] == "logistic_regression_latest", "kind"] == "latest value").all()
    assert not (out / "final").exists()   # nothing was saved from a final fit


def test_final_run_writes_report_tables_for_every_dataset(final_run):
    out, _ = final_run
    counts = pd.read_csv(out / "report" / "analysis_counts.csv").set_index("analysis")
    assert counts.loc["hospital_b", "patients"] == 70
    characteristics = pd.read_csv(out / "report" / "characteristics.csv")
    assert set(characteristics["dataset"]) == {"hospital_a_development", "hospital_a_holdout", "hospital_b"}
    shift = pd.read_csv(out / "report" / "characteristics_shift_a_vs_b.csv")
    assert {"hospital_a", "hospital_b", "standardized_difference"} <= set(shift.columns)
    age = shift[shift["variable"] == "age"]
    assert len(age) == 1


def test_final_run_saves_every_fitted_model_with_its_details(final_run):
    out, result = final_run
    thresholds = pd.read_csv(out / "dev" / "thresholds.csv").set_index("model")["threshold"]
    for name in MODELS:
        model, metadata = load_model(str(out / "final" / "artifacts" / "models" / name))
        assert metadata["model"] == name
        assert len(metadata["feature_columns"]) == model.n_features_in_
        assert metadata["threshold"] == pytest.approx(thresholds[name])
        assert metadata["trained_on"]["dataset"] == "hospital_a_development"
        assert metadata["trained_on"]["rows"] > 0 and metadata["seed"] == 7
        probabilities = model.predict_proba(np.zeros((3, model.n_features_in_)))
        assert probabilities.shape == (3, 2)
    assert any(f.endswith("gradient_boosting_with_missingness.joblib") for f in result["files"])


def test_final_run_saves_the_fitted_calibrators(final_run):
    out, _ = final_run
    folder = out / "final" / "artifacts" / "calibrators"
    grid = np.linspace(0.01, 0.99, 25)
    for name in MODELS:
        for variant in ("platt_dev", "isotonic_dev", "intercept_update_labeled_b"):
            path = folder / f"{name}_hour_{variant}.json"
            assert path.exists(), path.name
            calibrated = load_calibrator(str(path)).transform(grid)
            assert ((calibrated >= 0) & (calibrated <= 1)).all()
    # The development-fit calibrators are saved once; the intercept update comes from hospital B only.
    assert len(list(folder.glob("*_intercept_update_labeled_b.json"))) <= 2 * len(MODELS)



# ---------- settings and command line ----------

def test_settings_defaults_and_overrides():
    s = Settings.from_config({})
    assert (s.seed, s.n_splits, s.holdout_fraction, s.windows) == (42, 5, 0.2, (3, 6, 12))
    assert s.target_alerts_per_patient_day == 1.0 and s.onset_verified is False
    s = Settings.from_config(
        {"seed": 3, "bootstrap": {"n_resamples": 50}, "paths": {"results_dir": "r", "error_cases_table": "r/x.csv"}},
        n_resamples=5,
    )
    assert s.seed == 3 and s.n_resamples == 5 and s.review_path == "r/x.csv"
    moved = Settings.from_config({"paths": {"error_cases_table": "r/x.csv"}}, results_dir="elsewhere")
    assert moved.review_path == str(Path("elsewhere") / "manual_error_review.csv")


def test_model_specs_follow_the_missingness_setting():
    both = model_specs(Settings())
    assert set(both) == MODELS
    only_with = model_specs(Settings(missingness_variants=(True,)))
    assert "gradient_boosting_without_missingness" not in only_with


def test_unknown_main_model_is_rejected(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config_dict(tmp_path / "A", tmp_path / "B", pipeline={"main_model": "nope"})))
    with pytest.raises(ValueError, match="main_model"):
        run(str(cfg), results_dir=str(tmp_path / "out"), log=quiet)


def test_command_line_development_run(world, tmp_path, capsys):
    main(["--config", world, "--out", str(tmp_path / "cli"), "--n-resamples", "5"])
    out = capsys.readouterr().out
    assert "WARNING" in out and "done:" in out
    assert (tmp_path / "cli" / "dev" / "metrics.csv").exists()


def test_require_verified_onset_stops_the_run(world, tmp_path):
    with pytest.raises(SystemExit, match="onset_verified"):
        main(["--config", world, "--out", str(tmp_path / "cli"), "--require-verified-onset"])
    assert not (tmp_path / "cli").exists()


# ---------- recalibration helper ----------

def test_intercept_update_is_evaluated_only_on_patients_that_were_not_used_to_fit_it():
    rng = np.random.default_rng(0)
    dev = (np.repeat(np.arange(40), 5), (rng.random(200) < 0.3).astype(int), rng.random(200))
    ids = np.repeat(np.arange(100, 110), 4)
    p = np.repeat(np.arange(10) / 10, 4)                 # each patient's predictions are one distinct value
    y = (rng.random(40) < 0.4).astype(int)
    labeled = np.array([100, 101, 102])
    warnings = []
    rep = calibration_reports("m", dev, (ids, y, p), labeled, Settings(), warnings, "hospital_b")["hour"]
    uncal = rep[rep["variant"] == "uncalibrated"].iloc[0]
    assert uncal["mean_predicted"] == pytest.approx(p[~np.isin(ids, labeled)].mean())   # only the 7 evaluation patients
    assert "model" in rep.columns and rep["model"].iloc[0] == "m"

    no_update = calibration_reports("m", dev, (ids, y, p), None, Settings(), warnings, "holdout")["hour"]
    assert "intercept_update_labeled_b" not in set(no_update["variant"])
    assert no_update[no_update["variant"] == "uncalibrated"]["mean_predicted"].iloc[0] == pytest.approx(p.mean())
