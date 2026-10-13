# Study Design (write first, change only with a logged reason)

## 1. Question

At each ICU hour *t*, estimate the probability that a patient develops sepsis within the next *H* hours (default H = 6), using only information available at or before *t*.

## 2. Data

- Development: Challenge 2019 hospital A. External test: hospital B (never used for tuning).
- Unit of analysis: patient-hour. Unit of splitting and resampling: **patient**.

## 3. Labels and onset time

- The dataset `SepsisLabel` is derived from Sepsis-3 criteria (organ dysfunction + suspected infection).
- **Check the dataset documentation:** in the Challenge data the label turns 1 about 6 hours *before* the clinically defined onset. Reconstruct onset time explicitly (first label = 1 hour + shift), unit-test it, and document it here once verified.
- All early-vs-late analysis depends on this onset time, so get it right before anything else.

## 4. Splits (leakage control)

1. Split by patient, never by row. Use grouped K-fold on hospital A for tuning.
2. Keep a patient-level holdout from A that is touched once, at the end.
3. Hospital B is the external test: never used for tuning, thresholds, or recalibration fitting.
4. Fit all imputers, scalers, and calibrators on training data only.

## 5. Leakage guards

- Features at hour *t* use only data from hours <= *t*. Imputation is forward-fill only; no backward fill, no global means that include the future.
- Include an **ICU-length-of-stay-only baseline.** If it scores suspiciously well, the label or splits are leaking time information.
- Watch for features that are proxies for clinician suspicion (lab ordering patterns, missingness). They can inflate "early" performance without real physiological signal. Report results with and without missingness indicators.

## 6. Early vs. late signal

- Lead time = onset time minus time of first alert at a fixed alert threshold.
- Report performance by lead-time bin (pre-registered, adjustable with a logged reason):

| Bin | Definition |
|---|---|
| Early | alert >= 12h before onset |
| Intermediate | 6 to 12h before onset |
| Late | 0 to 6h before onset |
| Missed/post-onset | first alert after onset, or never |

- Also report the model score trajectory in the 24h before onset vs. matched non-sepsis controls.
- Report the Challenge utility score as a secondary operational metric.

## 7. Metrics

| Area | Metrics |
|---|---|
| Discrimination | AUROC, AUPRC (with prevalence stated) |
| Operating point | Sensitivity and PPV at a fixed alert rate; alerts per patient-day |
| Calibration | Reliability diagram, Brier score, calibration slope and intercept, ECE (secondary only; it depends on binning) |
| Uncertainty | 95% patient-level bootstrap CIs on everything |

Calibration is reported at both hour level and patient level, **before and after** recalibration. Recalibration (Platt or isotonic) is fit on development data only; for the external site, also report an intercept-only update fit on a small labeled portion of B and evaluated on the rest.

## 8. Baselines (the model must beat these, with CIs)

1. ICU-length-of-stay only
2. SIRS, qSOFA, NEWS2-style threshold scores
3. Logistic regression on latest values
4. Gradient boosting with rolling-window features (main model)

## 9. Error-mode analysis

- **False alarms:** alerts per patient-day, by ICU unit; what the high-scoring non-sepsis patients look like (chronic abnormal vitals, other infections, post-op).
- **Misses:** patients never alerted; patients alerted only after onset; common feature patterns.
- **Subgroups:** age, sex, unit, ICU stay length, amount of missing data; report sensitivity, PPV, and calibration for each.
- **Label noise:** note cases where the label itself looks questionable.
- **Shift:** performance and calibration on A vs. B.
- Keep a table of manually reviewed error cases in `results/`.

## 10. Reproducibility and reporting

- Fixed seeds, configs in `configs/`, results written by scripts (not by hand).
- Follow the TRIPOD+AI reporting checklist.
- Keep a decisions log below. Every change to this protocol gets a date and a reason.

## Decisions log

| Date | Change | Reason |
|---|---|---|
| | | |
