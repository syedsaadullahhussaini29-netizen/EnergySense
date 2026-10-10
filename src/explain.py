"""
explain.py
----------
Feature-importance utilities for EnergySense AI.

Uses sklearn-native importance (tree-based models) and
permutation importance as a model-agnostic fallback.

NOTE: Feature importance indicates how useful a feature is
for the model's predictions. It does NOT imply causality.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series,
                           n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['feature', 'importance', 'method'].

    Strategy
    --------
    1. If the model (or final estimator in a Pipeline) exposes
       `feature_importances_`, use those directly (fast, deterministic).
    2. Otherwise fall back to permutation importance on the test set.
    """
    # Unwrap pipeline if necessary
    estimator = model
    if hasattr(model, "named_steps"):
        # sklearn Pipeline — get last step
        last_step_name = list(model.named_steps.keys())[-1]
        estimator = model.named_steps[last_step_name]

    feature_names = list(X_test.columns)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        method = "Built-in (impurity-based)"
    else:
        # Permutation importance — slower but model-agnostic
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_root_mean_squared_error",
        )
        importances = perm.importances_mean
        method = "Permutation importance"

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "method": method,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
