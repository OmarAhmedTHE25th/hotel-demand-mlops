"""Check holdout isolation and shared-split reuse for ADR baselines."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.regression.train import fit_baselines, load_training_validation
from tests.test_splits import make_synthetic_data


def setup_data(tmp_path):
    data = make_synthetic_data(40)
    manifest = {
        "index_type": "zero-based row positions in the input dataframe",
        "train": list(range(28)), "val": list(range(28, 34)),
        "test": list(range(34, 40)),
    }
    path = tmp_path / "regression_indices.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "splits_dir": str(tmp_path), "adr_max": 500,
        "regression_exclude_features": [
            "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
        ],
    }
    return data, config, path


def test_uses_saved_positions_and_leaves_manifest_unchanged(tmp_path):
    data, config, path = setup_data(tmp_path)
    original = path.read_bytes()
    X_train, X_val, y_train, y_val = load_training_validation(data, config)
    assert X_train.index.tolist() == list(range(28))
    assert X_val.index.tolist() == list(range(28, 34))
    assert path.read_bytes() == original
    assert not {"ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"} & set(X_train)
    assert y_train.name == y_val.name == "ADR"


def test_rejects_overlapping_or_stale_manifest(tmp_path):
    data, config, path = setup_data(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["test"][0] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partition"):
        load_training_validation(data, config)


def test_validation_does_not_fit_imputer_or_encoder(tmp_path):
    data, config, _ = setup_data(tmp_path)
    data.loc[:27, "Children"] = np.nan
    data.loc[0, "Children"] = 2.0
    data.loc[28:33, "Children"] = 1000.0
    data.loc[28:33, "Meal"] = "validation_only"
    X_train, X_val, y_train, y_val = load_training_validation(data, config)
    fitted, metrics = fit_baselines(X_train, X_val, y_train, y_val, config)
    pipeline = fitted["linear_regression"]
    columns = pipeline.named_steps["preprocessor"].named_steps["columns"]
    numeric = columns.named_transformers_["numeric"]
    names = next(names for name, _, names in columns.transformers_ if name == "numeric")
    assert numeric.named_steps["imputer"].statistics_[names.index("Children")] == 2.0
    encoder = columns.named_transformers_["low_cardinality"].named_steps["encoder"]
    assert all("validation_only" not in categories for categories in encoder.categories_)
    assert metrics["split"].eq("validation").all()
    assert np.isfinite(metrics[["mae", "mse", "rmse", "r2"]].to_numpy()).all()
    np.testing.assert_allclose(
        fitted["dummy_mean"].predict(X_val), y_train.mean()
    )
    artifact_path = tmp_path / "linear.joblib"
    joblib.dump(pipeline, artifact_path)
    np.testing.assert_allclose(
        joblib.load(artifact_path).predict(X_val), pipeline.predict(X_val)
    )


def test_test_values_cannot_change_training_or_validation(tmp_path):
    data, config, _ = setup_data(tmp_path)
    first = load_training_validation(data, config)
    data.loc[34:, "ADR"] = 499.0
    data.loc[34:, "Meal"] = "test_only"
    second = load_training_validation(data, config)
    for before, after in zip(first, second):
        if isinstance(before, pd.DataFrame):
            pd.testing.assert_frame_equal(before, after)
        else:
            pd.testing.assert_series_equal(before, after)
