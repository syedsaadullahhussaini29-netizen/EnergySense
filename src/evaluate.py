"""
evaluate.py
-----------
Evaluation utilities for EnergySense AI models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """Return a dict with MAE, RMSE, and R² for a set of predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Accept a list of metric dicts (from compute_metrics) and return a
    sorted comparison DataFrame (ascending RMSE).
    """
    df = pd.DataFrame(results)
    df = df.sort_values("RMSE").reset_index(drop=True)
    return df


def select_best_model(results: list) -> str:
    """Return the model name with the lowest test-set RMSE."""
    best = min(results, key=lambda x: x["RMSE"])
    return best["model"]
