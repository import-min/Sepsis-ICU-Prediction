"""Patient-level splits and leakage guards (study_design.md, section 4).

Rules this module enforces:
  * Split by patient, never by row. Every row of a patient goes to one side.
  * A patient-level holdout from hospital A is carved out first and is not
    used for tuning. Touch it once, at the end.
  * Grouped K-fold (groups = patient) is used on the remaining development
    patients for tuning.
  * Splits are stratified by "ever septic" so every fold has septic patients.
  * Splits are deterministic: same patients + same seeds -> same split,
    regardless of row order.

Hospital B (the external test set) is never split or tuned on. Keep it out of
this module entirely, and use `assert_disjoint` to prove it shares no patients
with the development data.

Patient IDs must be unique across hospitals. If raw IDs can repeat between
hospital A and B, prefix them when loading (for example "A_p000001").

All functions take row-level arrays (one entry per patient-hour) and work on
unique patients internally, using numpy only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

ROLE_DEVELOPMENT = "development"
ROLE_HOLDOUT = "holdout"
HOLDOUT_FOLD = -1  # fold value for holdout patients (they are in no fold)


class PatientLeakageError(RuntimeError):
    """Raised when the same patient appears in two groups that must be separate."""


def patient_outcomes(
    patient_ids: Sequence, labels: Sequence
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse row-level data to one entry per patient.

    Returns (unique_patient_ids sorted, ever_septic bool array), where
    ever_septic is True if the patient has any row with SepsisLabel == 1.
    """
    ids = np.asarray(patient_ids)
    y = np.asarray(labels)
    if ids.ndim != 1 or ids.shape != y.shape:
        raise ValueError("patient_ids and labels must be 1-D arrays of equal length")
    if ids.size == 0:
        raise ValueError("no rows given")

    unique, inverse = np.unique(ids, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    ever = np.zeros(unique.size, dtype=bool)
    ever[inverse[y == 1]] = True
    return unique, ever


@dataclass(frozen=True, eq=False)
class PatientSplit:
    """Role and fold for every unique patient."""

    patient_ids: np.ndarray  # unique, sorted
    ever_septic: np.ndarray  # bool
    role: np.ndarray  # ROLE_DEVELOPMENT or ROLE_HOLDOUT
    fold: np.ndarray  # 0..n_splits-1 for development, HOLDOUT_FOLD for holdout
    n_splits: int

    def holdout_ids(self) -> np.ndarray:
        return self.patient_ids[self.role == ROLE_HOLDOUT]

    def development_ids(self) -> np.ndarray:
        return self.patient_ids[self.role == ROLE_DEVELOPMENT]

    def train_val_ids(self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """(train_ids, validation_ids) for one fold. Holdout patients are in neither."""
        if not 0 <= fold < self.n_splits:
            raise ValueError(f"fold must be in [0, {self.n_splits - 1}]")
        dev = self.role == ROLE_DEVELOPMENT
        train = self.patient_ids[dev & (self.fold != fold)]
        val = self.patient_ids[dev & (self.fold == fold)]
        return train, val


def make_patient_split(
    patient_ids: Sequence,
    labels: Sequence,
    holdout_fraction: float = 0.2,
    n_splits: int = 5,
    holdout_seed: int = 42,
    fold_seed: int = 42,
) -> PatientSplit:
    """Carve out a patient-level holdout, then assign development patients to folds.

    Both steps are stratified by ever-septic status.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1 (exclusive)")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    ids, ever = patient_outcomes(patient_ids, labels)
    n = ids.size
    role = np.full(n, ROLE_DEVELOPMENT)

    # 1) Holdout, stratified.
    rng = np.random.default_rng(holdout_seed)
    for stratum in (True, False):
        idx = np.flatnonzero(ever == stratum)
        if idx.size == 0:
            continue
        n_hold = int(round(holdout_fraction * idx.size))
        if n_hold < 1 or n_hold >= idx.size:
            name = "septic" if stratum else "non-septic"
            raise ValueError(
                f"holdout_fraction={holdout_fraction} leaves no {name} patients on one "
                f"side ({idx.size} {name} patients in total)"
            )
        rng.shuffle(idx)
        role[idx[:n_hold]] = ROLE_HOLDOUT

    # 2) Folds among development patients, stratified. The offset keeps total
    #    fold sizes within one patient of each other.
    fold = np.full(n, HOLDOUT_FOLD, dtype=int)
    dev = role == ROLE_DEVELOPMENT
    rng = np.random.default_rng(fold_seed)
    offset = 0
    for stratum in (True, False):
        idx = np.flatnonzero(dev & (ever == stratum))
        if 0 < idx.size < n_splits:
            name = "septic" if stratum else "non-septic"
            raise ValueError(
                f"only {idx.size} {name} development patients for {n_splits} folds"
            )
        rng.shuffle(idx)
        fold[idx] = (np.arange(idx.size) + offset) % n_splits
        offset += idx.size

    split = PatientSplit(
        patient_ids=ids,
        ever_septic=ever,
        role=role,
        fold=fold,
        n_splits=n_splits,
    )
    assert_disjoint(development=split.development_ids(), holdout=split.holdout_ids())
    return split


def rows_in(row_patient_ids: Sequence, selected_ids: Sequence) -> np.ndarray:
    """Boolean mask over rows: True where the row's patient is in selected_ids."""
    return np.isin(np.asarray(row_patient_ids), np.asarray(selected_ids))


def assert_disjoint(**groups: Sequence) -> None:
    """Raise PatientLeakageError if any patient appears in more than one group.

    Example:
        assert_disjoint(development=dev_ids, hospital_b=b_ids)
    """
    names = list(groups)
    sets = {name: set(np.asarray(groups[name]).tolist()) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sets[a] & sets[b]
            if common:
                example = sorted(common, key=str)[:3]
                raise PatientLeakageError(
                    f"{len(common)} patient(s) appear in both '{a}' and '{b}', "
                    f"for example {example}"
                )


def save_split(path: str, split: PatientSplit) -> None:
    """Write the split to CSV so the exact assignment can be audited.

    Columns: patient_id, ever_septic, role, fold (-1 for holdout).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "ever_septic", "role", "fold"])
        for pid, ever, role, fold in zip(
            split.patient_ids, split.ever_septic, split.role, split.fold
        ):
            writer.writerow([str(pid), int(ever), str(role), int(fold)])
