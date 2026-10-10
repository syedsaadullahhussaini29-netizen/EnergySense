"""
train.py
--------
End-to-end model training pipeline for EnergySense AI.

Run from the project root:
    python src/train.py

Outputs (saved to models/):
    best_model.joblib      — the trained model with the lowest test RMSE
    model_metrics.json     — metrics for all trained models + best model name
    feature_columns.json   — list of feature column names used by the model

Train/test strategy
-------------------
Because this is a time-series dataset (10-minute intervals, Jan–May 2016),
we use a chronological split: the earliest 80% of observations are used for
training and the latest 20% for testing. This avoids temporal data leakage and
produces a more realistic estimate of generalisation performance.

Lag features (past Appliances values) are engineered before the split so that
no future information bleeds into the model.
"""

import os
import sys
import json
import time

import joblib
import numpy as np

# Add project root to path so local imports work when run directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.preprocessing import load_and_clean, DATA_PATH
from src.feature_engineering import engineer_features, build_X_y, get_feature_columns
from src.evaluate import compute_metrics, compare_models, select_best_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_RATIO = 0.20   # last 20 % of chronological observations as test set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chronological_split(X, y, test_ratio: float = TEST_RATIO):
    """
    Split preserving temporal order — no shuffling.
    Earlier observations -> training; later observations -> test.
    """
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:],
    )


def build_models() -> dict:
    """
    Return a dict of {name: sklearn estimator / Pipeline}.

    Hyperparameters are tuned for this dataset's temporal characteristics:
    - Lag-1 (10-min) is the dominant feature (importance ~0.60).
    - Tree-based models need regularisation (shallow depth, min_samples_leaf)
      to avoid memorising training patterns that don't generalise.
    - Linear Regression benefits from StandardScaler (scale-sensitive).
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    # Optionally include XGBoost if installed
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        print("XGBoost found — including in model comparison.")
    except ImportError:
        print("XGBoost not installed — skipping.")

    return models


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- 1. Load & clean ---------------------------------------------------
    print("Loading dataset ...")
    df = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns.")

    # ---- 2. Feature engineering --------------------------------------------
    print("Engineering features (with lag features) ...")
    df = engineer_features(df, include_lags=True)
    feature_cols = get_feature_columns(df)
    X, y = build_X_y(df)
    print(f"  Usable rows after lag warm-up: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target range: [{int(y.min())} - {int(y.max())}] Wh")

    # ---- 3. Chronological train / test split --------------------------------
    print(f"Splitting data (chronological, test={int(TEST_RATIO*100)}%) ...")
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # ---- 4. Train & evaluate models ----------------------------------------
    models = build_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_metrics = compute_metrics(y_test, y_pred_test, model_name=name)
        train_r2 = float(np.mean(
            1 - np.sum((y_train - y_pred_train) ** 2) /
            np.sum((y_train - y_train.mean()) ** 2)
        ))

        test_metrics["train_R2"] = round(train_r2, 4)
        test_metrics["train_time_s"] = round(elapsed, 2)
        results.append(test_metrics)
        trained_models[name] = model

        print(
            f"  MAE={test_metrics['MAE']:.2f}  "
            f"RMSE={test_metrics['RMSE']:.2f}  "
            f"R2={test_metrics['R2']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

    # ---- 5. Select best model ----------------------------------------------
    best_name = select_best_model(results)
    best_model = trained_models[best_name]
    print(f"\nBest model by test RMSE: {best_name}")

    comparison_df = compare_models(results)
    print("\nModel comparison:")
    print(comparison_df[["model", "MAE", "RMSE", "R2"]].to_string(index=False))

    # ---- 6. Save artifacts -------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nSaved model -> {model_path}")

    metrics_dict = {
        "best_model": best_name,
        "test_ratio": TEST_RATIO,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(feature_cols),
        "models": results,
    }
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    feature_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)
    print(f"Saved features -> {feature_path}")

    print("\nTraining complete.")
    return best_model, metrics_dict, feature_cols


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
