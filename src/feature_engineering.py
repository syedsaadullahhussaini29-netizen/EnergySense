"""
feature_engineering.py
-----------------------
Creates temporal and lag features for the EnergySense AI model.

All features are derived purely from existing columns — no future information
or target-variable leakage. Lag features use past target values only
(shifted so they cannot see the current/future target).
"""

import pandas as pd
import numpy as np

DATE_COL = "date"
TARGET_COL = "Appliances"

# Lag steps (in 10-minute intervals)
# Lag-1 = 10 min ago, Lag-6 = 1 hour ago, Lag-144 = 24 hours ago
LAG_STEPS = [1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 144]

# Rolling window sizes (in 10-minute intervals)
ROLLING_WINDOWS = [6, 12, 24, 36, 144]


# ---------------------------------------------------------------------------
# Part-of-day helper
# ---------------------------------------------------------------------------

def _part_of_day(hour: pd.Series) -> pd.Series:
    """Return a string label for the part of the day."""
    conditions = [
        hour.between(5, 8),
        hour.between(9, 12),
        hour.between(13, 17),
        hour.between(18, 21),
    ]
    choices = ["Morning", "Late Morning", "Afternoon", "Evening"]
    return np.select(conditions, choices, default="Night")


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """
    Accept a clean DataFrame (output of load_and_clean) and return a new
    DataFrame with all original columns plus engineered features.

    Temporal features
    -----------------
    hour, minute, day_of_week, month, day : basic date/time parts
    is_weekend       : 1 if Saturday/Sunday, else 0
    part_of_day      : categorical label
    hour_sin/cos     : cyclical encoding of hour
    dow_sin/cos      : cyclical encoding of day-of-week
    month_sin/cos    : cyclical encoding of month

    Lag features (if include_lags=True)
    ------------------------------------
    appliances_lag_N  : Appliances value N steps (10-min intervals) ago
    appliances_roll_mean_W : Rolling mean over W steps
    appliances_roll_std_W  : Rolling std over W steps
    """
    df = df.copy()

    dt = df[DATE_COL]

    # ---- Temporal features ----
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek        # 0 = Monday
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["part_of_day"] = _part_of_day(df["hour"])

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- Lag features ----
    if include_lags:
        for lag in LAG_STEPS:
            df[f"appliances_lag_{lag}"] = df[TARGET_COL].shift(lag)

        for window in ROLLING_WINDOWS:
            rolled = df[TARGET_COL].shift(1).rolling(window)
            df[f"appliances_roll_mean_{window}"] = rolled.mean()
            df[f"appliances_roll_std_{window}"] = rolled.std()

        # Drop rows where lags are NaN (first LAG_STEPS[-1] rows)
        df = df.dropna().reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Build feature matrix and target vector
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of input feature column names used for modelling.
    Excludes the date column, target column, and categorical 'part_of_day'.
    """
    exclude = {DATE_COL, TARGET_COL, "part_of_day"}
    return [c for c in df.columns if c not in exclude]


def build_X_y(df: pd.DataFrame):
    """
    Return (X, y) where X is the feature DataFrame and y is the target Series.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ---------------------------------------------------------------------------
# Feature engineering for a single prediction row (no lag NaN drop)
# ---------------------------------------------------------------------------

def engineer_features_for_prediction(row_dict: dict) -> pd.DataFrame:
    """
    Given a dict of raw feature values (for the What-If Simulator and
    Prediction page), return a one-row DataFrame with all temporal features
    added. Lag features are passed in directly via row_dict.
    """
    df_row = pd.DataFrame([row_dict])
    if DATE_COL in df_row.columns:
        dt = pd.to_datetime(df_row[DATE_COL])
        df_row["hour"] = dt.dt.hour
        df_row["minute"] = dt.dt.minute
        df_row["day_of_week"] = dt.dt.dayofweek
        df_row["month"] = dt.dt.month
        df_row["day"] = dt.dt.day
        df_row["is_weekend"] = (df_row["day_of_week"] >= 5).astype(int)
        df_row["hour_sin"] = np.sin(2 * np.pi * df_row["hour"] / 24)
        df_row["hour_cos"] = np.cos(2 * np.pi * df_row["hour"] / 24)
        df_row["dow_sin"] = np.sin(2 * np.pi * df_row["day_of_week"] / 7)
        df_row["dow_cos"] = np.cos(2 * np.pi * df_row["day_of_week"] / 7)
        df_row["month_sin"] = np.sin(2 * np.pi * df_row["month"] / 12)
        df_row["month_cos"] = np.cos(2 * np.pi * df_row["month"] / 12)
    return df_row
