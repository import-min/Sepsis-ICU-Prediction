"""Unit tests for patient-level splits and leakage guards.

Run from the repo root:  pytest -q
"""

import csv
import sys
from pathlib import Path

# Lets `pytest` find src/ without installing the package.
# Remove once the repo has a pyproject.toml and you run `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from sepsis_icu.splits import (
    HOLDOUT_FOLD,
    ROLE_DEVELOPMENT,
    ROLE_HOLDOUT,
    PatientLeakageError,
    assert_disjoint,
    make_patient_split,
    patient_outcomes,
    rows_in,
    save_split,
)


def make_rows(n_patients=200, n_septic=20, prefix="A", seed=0):
    """Row-level data: several hours per patient, rows shuffled."""
    rng = np.random.default_rng(seed)
    ids, labels = [], []
    for i in range(n_patients):
        length = int(rng.integers(10, 40))
        y = np.zeros(length, dtype=int)
        if i < n_septic:
            y[int(rng.integers(0, length)):] = 1
        ids += [f"{prefix}_p{i:06d}"] * length
        labels += y.tolist()
    ids, labels = np.array(ids), np.array(labels)
    perm = rng.permutation(ids.size)
    return ids[perm], labels[perm]


# ---------- patient_outcomes ----------

def test_patient_outcomes_collapses_rows():
    ids = ["b", "a", "a", "b", "c", "c"]
    labels = [0, 0, 1, 0, 0, 0]
    unique, ever = patient_outcomes(ids, labels)
    assert list(unique) == ["a", "b", "c"]
    assert list(ever) == [True, False, False]


def test_patient_outcomes_rejects_bad_shapes():
    with pytest.raises(ValueError):
        patient_outcomes(["a", "b"], [0])
    with pytest.raises(ValueError):
        patient_outcomes([], [])


# ---------- holdout ----------

def test_holdout_is_stratified_and_sized_correctly():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, holdout_fraction=0.2)
    hold = split.role == ROLE_HOLDOUT
    assert (hold & split.ever_septic).sum() == 4      # 20% of 20 septic
    assert (hold & ~split.ever_septic).sum() == 36    # 20% of 180 non-septic


def test_holdout_and_development_share_no_patients():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels)
    assert_disjoint(development=split.development_ids(), holdout=split.holdout_ids())
    assert len(split.development_ids()) + len(split.holdout_ids()) == 200


# ---------- folds ----------

def test_every_development_patient_is_validated_exactly_once():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, n_splits=5)
    seen = []
    for k in range(5):
        _, val = split.train_val_ids(k)
        seen.extend(val.tolist())
    assert sorted(seen) == sorted(split.development_ids().tolist())
    assert len(seen) == len(set(seen))


def test_holdout_patients_are_in_no_fold():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, n_splits=5)
    holdout = set(split.holdout_ids().tolist())
    assert (split.fold[split.role == ROLE_HOLDOUT] == HOLDOUT_FOLD).all()
    for k in range(5):
        train, val = split.train_val_ids(k)
        assert not holdout & set(train.tolist())
        assert not holdout & set(val.tolist())


def test_folds_are_balanced_in_size_and_septic_count():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, n_splits=5)
    dev = split.role == ROLE_DEVELOPMENT
    sizes = [int(((split.fold == k) & dev).sum()) for k in range(5)]
    septic = [int(((split.fold == k) & dev & split.ever_septic).sum()) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert max(septic) - min(septic) <= 1
    assert min(septic) >= 1


def test_row_level_masks_never_split_a_patient():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, n_splits=5)
    holdout_rows = rows_in(ids, split.holdout_ids())
    for k in range(5):
        train, val = split.train_val_ids(k)
        train_rows = rows_in(ids, train)
        val_rows = rows_in(ids, val)
        # Every row is in exactly one of train / validation / holdout.
        assert (train_rows.astype(int) + val_rows + holdout_rows == 1).all()
        # No patient has rows in both train and validation.
        assert not set(ids[train_rows].tolist()) & set(ids[val_rows].tolist())


def test_train_val_ids_rejects_bad_fold():
    ids, labels = make_rows()
    split = make_patient_split(ids, labels, n_splits=5)
    with pytest.raises(ValueError):
        split.train_val_ids(5)
    with pytest.raises(ValueError):
        split.train_val_ids(-1)


# ---------- determinism ----------

def test_split_is_deterministic_and_independent_of_row_order():
    ids, labels = make_rows()
    a = make_patient_split(ids, labels, holdout_seed=1, fold_seed=2)
    perm = np.random.default_rng(99).permutation(ids.size)
    b = make_patient_split(ids[perm], labels[perm], holdout_seed=1, fold_seed=2)
    assert (a.patient_ids == b.patient_ids).all()
    assert (a.role == b.role).all()
    assert (a.fold == b.fold).all()


def test_different_seeds_give_different_splits():
    ids, labels = make_rows()
    a = make_patient_split(ids, labels, holdout_seed=1)
    b = make_patient_split(ids, labels, holdout_seed=2)
    assert not (a.role == b.role).all()


# ---------- input validation ----------

@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_bad_holdout_fraction_raises(fraction):
    ids, labels = make_rows()
    with pytest.raises(ValueError):
        make_patient_split(ids, labels, holdout_fraction=fraction)


def test_too_few_splits_raises():
    ids, labels = make_rows()
    with pytest.raises(ValueError):
        make_patient_split(ids, labels, n_splits=1)


def test_too_few_septic_patients_for_folds_raises():
    ids, labels = make_rows(n_patients=40, n_septic=4)
    with pytest.raises(ValueError, match="septic"):
        make_patient_split(ids, labels, n_splits=5)


def test_holdout_that_empties_a_stratum_raises():
    ids, labels = make_rows(n_patients=10, n_septic=2)  # 20% of 2 rounds to 0
    with pytest.raises(ValueError, match="holdout_fraction"):
        make_patient_split(ids, labels, holdout_fraction=0.2, n_splits=2)


# ---------- assert_disjoint ----------

def test_assert_disjoint_passes_for_separate_groups():
    assert_disjoint(development=["a", "b"], hospital_b=["c", "d"])


def test_assert_disjoint_catches_overlap():
    with pytest.raises(PatientLeakageError, match="hospital_b"):
        assert_disjoint(development=["a", "b"], hospital_b=["b", "c"])


def test_assert_disjoint_checks_every_pair():
    with pytest.raises(PatientLeakageError):
        assert_disjoint(a=["x"], b=["y"], c=["y"])


def test_hospital_prefix_prevents_id_collisions():
    # Same raw IDs in both hospitals look like leakage until prefixed.
    raw = ["p000001", "p000002"]
    with pytest.raises(PatientLeakageError):
        assert_disjoint(development=raw, hospital_b=raw)
    assert_disjoint(
        development=[f"A_{p}" for p in raw],
        hospital_b=[f"B_{p}" for p in raw],
    )


# ---------- save_split ----------

def test_save_split_writes_auditable_csv(tmp_path):
    ids, labels = make_rows()
    split = make_patient_split(ids, labels)
    out = tmp_path / "results" / "split.csv"
    save_split(str(out), split)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200
    assert set(rows[0]) == {"patient_id", "ever_septic", "role", "fold"}
    assert {r["role"] for r in rows} == {ROLE_DEVELOPMENT, ROLE_HOLDOUT}
    assert all(int(r["fold"]) == HOLDOUT_FOLD for r in rows if r["role"] == ROLE_HOLDOUT)
    assert all(0 <= int(r["fold"]) < 5 for r in rows if r["role"] == ROLE_DEVELOPMENT)
