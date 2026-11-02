"""Milestone 1 ADR baselines using shared preprocessing and saved splits."""

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.cleaning import clean_data, regression_filter
from src.data.loader import load_config, load_h2
from src.features.feature_sets import get_features
from src.preprocessing.pipeline import build_preprocessor


def load_training_validation(data, config):
    """Read the team's manifest without regenerating splits or selecting test data.

    Positions refer to the cleaned, ADR-filtered frame, not the raw CSV.
    Validate the whole partition structurally, but select only train/validation.
    """
    population = regression_filter(data, float(config.get("adr_max", 500)))
    path = Path(config.get("splits_dir", "data/splits")) / "regression_indices.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("index_type") != "zero-based row positions in the input dataframe":
        raise ValueError("Unrecognized shared split index convention.")
    groups = [manifest[name] for name in ("train", "val", "test")]
    if any(not group for group in groups):
        raise ValueError("Every shared split must be nonempty.")
    positions = [position for group in groups for position in group]
    if any(type(position) is not int for position in positions):
        raise ValueError("Shared split positions must be integers.")
    if sorted(positions) != list(range(len(population))):
        raise ValueError("Shared splits must partition the filtered data exactly; check data/version.")
    features = get_features("regression", config)
    if "ADR" in features:
        raise ValueError("ADR must never be an input feature.")
    train = population.iloc[manifest["train"]]
    validation = population.iloc[manifest["val"]]
    return (
        train.loc[:, features], validation.loc[:, features],
        train["ADR"], validation["ADR"],
    )


def fit_baselines(X_train, X_val, y_train, y_val, config):
    """Fit a fresh complete pipeline for each model; evaluate validation only."""
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
    }
    fitted, rows = {}, []
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor("regression", config, scale=True)),
            ("model", model),
        ])
        started = perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
        predictions = pipeline.predict(X_val)
        mse = float(mean_squared_error(y_val, predictions))
        rows.append({
            "model": name, "split": "validation",
            "mae": float(mean_absolute_error(y_val, predictions)),
            "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_val, predictions)),
            "fit_seconds": fit_seconds,
        })
        fitted[name] = pipeline
    return fitted, pd.DataFrame(rows)


def train_regression():
    """Run reproducible, untuned ADR experiments without test evaluation."""
    config = load_config()
    raw_path = Path(config.get("data_path", "data/H2.csv"))
    # Reuse the teammate's deterministic cleaning, preserving row order.
    cleaned = clean_data(load_h2(raw_path))
    X_train, X_val, y_train, y_val = load_training_validation(cleaned, config)
    fitted, metrics = fit_baselines(X_train, X_val, y_train, y_val, config)
    output = Path(config.get("outputs_dir", "outputs")) / "regression"
    artifacts = Path(config.get("artifacts_dir", "artifacts"))
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    for name, pipeline in fitted.items():
        joblib.dump(pipeline, artifacts / f"regression_{name}.joblib")
    split_path = Path(config.get("splits_dir", "data/splits")) / "regression_indices.json"
    metadata = {
        "evaluation_split": "validation", "test_evaluated": False,
        "train_rows": len(X_train), "validation_rows": len(X_val),
        "features": list(X_train.columns), "config": config,
        "data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("Final test set was not selected, transformed, predicted, or scored.")
    print(f"Results: {output}; fitted pipelines: {artifacts}")
