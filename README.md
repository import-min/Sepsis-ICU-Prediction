# Sepsis Prediction in ICU Patients: Early vs. Late Signal, Calibration, and Error Modes

> Research project. **Not a medical device. Not for clinical use.**

## Goal

Build an hourly sepsis-risk model for ICU patients and evaluate it honestly. The aim is not just a high AUROC, but answers to three questions:

1. **Early vs. late signal.** How far ahead of sepsis onset does the model actually fire? Does performance hold at 12h, 6h, and 0h before onset, or does it only work once clinicians already suspect sepsis?
2. **Calibration.** When the model says 20% risk, is it 20%? Does that survive moving to a different hospital?
3. **Error modes.** Who gets false alarms, who gets missed, and why?

The full evaluation protocol is in [`docs/study_design.md`](docs/study_design.md). It is written **before** modeling so results cannot quietly shape the rules.

## Status

- [x] Study design written
- [ ] Data downloaded and loaded (see [`data/README.md`](data/README.md))
- [ ] Label + onset reconstruction, with tests
- [ ] Patient-level splits (development hospital / external hospital)
- [ ] Baselines (clinical scores, ICU-length-of-stay only, logistic regression)
- [ ] Main model (gradient boosting)
- [ ] Early-vs-late analysis
- [ ] Calibration analysis (+ recalibration at the external site)
- [ ] Error-mode analysis
- [ ] Write-up / model card

## Repository layout

```
data/        Data instructions only. Raw data is NEVER committed.
docs/        Study design, decisions log
src/sepsis/  Python package (labels, splits, features, models, evaluation)
notebooks/   Exploration (added later)
configs/     Experiment configs (added later)
results/     Metrics tables and figures (added later)
tests/       Leakage and label tests (added later)
```

## Quickstart

```bash
git clone <your-repo-url>
cd sepsis-icu-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to get the data.

## Results

_To be filled in. Every reported number gets a 95% patient-level bootstrap confidence interval._

## Limitations

_To be filled in as the project goes. Known from the start: sepsis labels derived from Sepsis-3 criteria are proxies for the clinical truth, not the truth itself._
