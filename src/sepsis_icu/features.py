"""Leakage-safe features (study_design.md, sections 4 and 5).

Rules this module enforces:
  * Features at hour t use only rows from hours <= t. Nothing looks forward.
  * Imputation is forward-fill only, within each patient. No backward fill,
    no values carried across patients, no means that include the future.
  * Any constant used to fill what is left (medians) is fit on training
    patients only, then applied to other data.
  * Missingness indicators can be switched on or off, so results can be
    reported with and without them (they can act as proxies for clinician
    suspicion, for example how often a lab is ordered).

Assumption: rows are consecutive hourly rows per patient (as in the Challenge
2019 data), so "w rows back" means "w hours back". `build_features` checks this
and raises if it is false, so you find out instead of silently getting wrong
windows.

Whether to also give the model the time column (ICULOS) as a feature is a
modeling choice. It is kept in the output so you can build the ICU-length-of-
stay-only baseline from it, but it is not turned into extra features here.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

DEFAULT_ID_COLUMN = "patient_id"
DEFAULT_TIME_COLUMN = "ICULOS"
DEFAULT_WINDOWS = (3, 6, 12)  # hours; matches configs/default.yaml


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _assert_sorted(df: pd.DataFrame, id_col: str, time_col: str) -> None:
    keys = df[[id_col, time_col]].reset_index(drop=True)
    ordered = keys.sort_values([id_col, time_col], kind="stable").reset_index(drop=True)
    if not keys.equals(ordered):
        raise ValueError(
            "rows must be sorted by (patient, time) before forward-filling; "
            "otherwise values could be filled from the future"
        )


def _assert_unique_and_hourly(
    df: pd.DataFrame, id_col: str, time_col: str, require_hourly: bool
) -> None:
    if df.duplicated([id_col, time_col]).any():
        raise ValueError("duplicate (patient, time) rows")
    if require_hourly:
        step = df.groupby(id_col, sort=False)[time_col].diff().dropna()
        if (step != 1).any():
            raise ValueError(
                "rows are not consecutive hourly rows for every patient; "
                "window features assume 1 row = 1 hour "
                "(pass require_hourly=False to skip this check)"
            )


def forward_fill(
    df: pd.DataFrame,
    columns: Sequence[str],
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.DataFrame:
    """Forward-fill `columns` within each patient. Never fills backward.

    Leading missing values stay missing. Values are never carried from one
    patient to the next. Input must already be sorted by (patient, time).
    """
    _require_columns(df, [id_col, time_col, *columns])
    _assert_sorted(df, id_col, time_col)
    return df.groupby(id_col, sort=False)[list(columns)].ffill()


def hours_since_measured(
    df: pd.DataFrame,
    column: str,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.Series:
    """Hours since `column` was last actually measured (0 in an hour it was measured).

    Missing until the first measurement. Uses only the current and earlier rows.
    """
    t = df[time_col].astype(float)
    last_seen = t.where(df[column].notna()).groupby(df[id_col], sort=False).ffill()
    return t - last_seen


def build_features(
    df: pd.DataFrame,
    value_columns: Sequence[str],
    windows: Sequence[int] = DEFAULT_WINDOWS,
    missingness_indicators: bool = True,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
    require_hourly: bool = True,
) -> pd.DataFrame:
    """Build features for every patient-hour from the hours up to and including it.

    Returns a DataFrame sorted by (patient, time) with columns:
      id_col, time_col
      {col}_ffill        last known value (forward-filled)
      {col}_mean{w}h     mean of the last w hours of the forward-filled value
      {col}_min{w}h      min over the last w hours
      {col}_max{w}h      max over the last w hours
      {col}_delta{w}h    value now minus value w hours ago (missing if < w hours of history)
    and, if missingness_indicators is True:
      {col}_observed     1 if the value was measured in this hour, else 0
      {col}_hours_since  hours since the value was last measured

    Labels are not included; join them back on (id_col, time_col).
    """
    value_columns = list(value_columns)
    windows = [int(w) for w in windows]
    if not value_columns:
        raise ValueError("value_columns is empty")
    if any(w < 1 for w in windows):
        raise ValueError("windows must be positive whole numbers of hours")
    _require_columns(df, [id_col, time_col, *value_columns])

    df = df.sort_values([id_col, time_col], kind="stable").reset_index(drop=True)
    _assert_unique_and_hourly(df, id_col, time_col, require_hourly)

    ids = df[id_col]
    ff = forward_fill(df, value_columns, id_col, time_col)
    grouped = ff.groupby(ids, sort=False)

    feats = {}
    for col in value_columns:
        feats[f"{col}_ffill"] = ff[col]

    for w in windows:
        for how in ("mean", "min", "max"):
            rolled = getattr(grouped.rolling(window=w, min_periods=1), how)()
            rolled = rolled.reset_index(level=0, drop=True).reindex(df.index)
            for col in value_columns:
                feats[f"{col}_{how}{w}h"] = rolled[col]
        past = grouped.shift(w)
        for col in value_columns:
            feats[f"{col}_delta{w}h"] = ff[col] - past[col]

    if missingness_indicators:
        for col in value_columns:
            feats[f"{col}_observed"] = df[col].notna().astype(int)
            feats[f"{col}_hours_since"] = hours_since_measured(df, col, id_col, time_col)

    return pd.concat([df[[id_col, time_col]], pd.DataFrame(feats, index=df.index)], axis=1)


def fit_medians(
    train_features: pd.DataFrame,
    id_col: str = DEFAULT_ID_COLUMN,
    time_col: str = DEFAULT_TIME_COLUMN,
) -> pd.Series:
    """Median of each feature column, from TRAINING patients only."""
    cols: List[str] = [c for c in train_features.columns if c not in (id_col, time_col)]
    return train_features[cols].median()


def apply_medians(features: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill remaining missing values with medians fit on training data.

    Optional: tree models such as gradient boosting can handle missing values
    directly; logistic regression cannot. Returns a copy.
    """
    return features.fillna(medians)
