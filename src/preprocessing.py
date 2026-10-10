"""
preprocessing.py
----------------
Dataset loading, validation, and cleaning pipeline for EnergySense AI.

Dataset: UCI Appliances Energy Prediction
  - 19,735 observations at 10-minute intervals (Jan–May 2016)
  - 29 columns: date, Appliances (target), lights, 9 indoor temp/humidity pairs,
    outdoor weather variables, and two random noise columns (rv1, rv2).
  - No missing values; no duplicate rows.
"""

import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "energydata_complete.csv")
TARGET_COL = "Appliances"
DATE_COL = "date"

# Random noise columns — not useful for prediction
NOISE_COLS = ["rv1", "rv2"]

EXPECTED_COLS = [
    "date", "Appliances", "lights",
    "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
    "T4", "RH_4", "T5", "RH_5", "T6", "RH_6",
    "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed",
    "Visibility", "Tdewpoint", "rv1", "rv2",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(path: str = DATA_PATH) -> dict:
    """
    Load the CSV and return a validation report dict.
    Raises FileNotFoundError if the file is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place 'energydata_complete.csv' in the data/ folder."
        )

    df_raw = pd.read_csv(path)

    missing_per_col = df_raw.isnull().sum().to_dict()
    total_missing = int(sum(missing_per_col.values()))
    duplicates = int(df_raw.duplicated().sum())

    # Date parsing
    date_parse_ok = False
    date_min = date_max = None
    try:
        dates = pd.to_datetime(df_raw[DATE_COL])
        date_parse_ok = True
        date_min = str(dates.min())
        date_max = str(dates.max())
    except Exception:
        pass

    # Missing expected columns
    missing_cols = [c for c in EXPECTED_COLS if c not in df_raw.columns]
    extra_cols = [c for c in df_raw.columns if c not in EXPECTED_COLS]

    report = {
        "rows": int(df_raw.shape[0]),
        "columns": int(df_raw.shape[1]),
        "column_names": list(df_raw.columns),
        "missing_expected_columns": missing_cols,
        "extra_columns": extra_cols,
        "total_missing_values": total_missing,
        "missing_per_column": missing_per_col,
        "duplicate_rows": duplicates,
        "date_parse_ok": date_parse_ok,
        "date_range_start": date_min,
        "date_range_end": date_max,
        "target_column": TARGET_COL,
        "target_min": float(df_raw[TARGET_COL].min()),
        "target_max": float(df_raw[TARGET_COL].max()),
        "target_mean": float(df_raw[TARGET_COL].mean()),
        "target_median": float(df_raw[TARGET_COL].median()),
    }
    return report


# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV, parse the date column, sort chronologically,
    and drop random-noise columns.
    Returns a clean DataFrame.
    """
    df = pd.read_csv(path)

    # Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Sort chronologically (dataset is already sorted but ensure it)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Drop noise columns — they carry no predictive signal
    df = df.drop(columns=[c for c in NOISE_COLS if c in df.columns])

    # Ensure numeric columns are float where expected
    numeric_cols = [c for c in df.columns if c != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
