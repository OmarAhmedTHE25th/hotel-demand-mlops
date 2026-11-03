"""Model definitions and factories for Task B classification."""

from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_seed: int = 42) -> Dict[str, Any]:
    """Return dictionary of untuned baseline classifiers."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=random_seed,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def get_search_spaces() -> Dict[str, Dict[str, Any]]:
    """Return hyperparameter search ranges for tuning."""
    return {
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"],
        },
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },
    }