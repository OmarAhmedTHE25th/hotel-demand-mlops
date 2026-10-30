"""Tests for the train-fitted preprocessing pipeline."""

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_sets import get_features
from src.preprocessing.pipeline import build_preprocessor


def make_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train and validation rows with distinct validation categories."""
    config: dict[str, Any] = {"classification_use_adr": False}
    columns = get_features("classification", config)
    train = pd.DataFrame(index=range(3), columns=columns)
    validation = pd.DataFrame(index=range(2), columns=columns)

    categorical = {
        "ArrivalDateMonth",
        "Meal",
        "MarketSegment",
        "DistributionChannel",
        "ReservedRoomType",
        "DepositType",
        "CustomerType",
        "Country",
        "Agent",
    }
    for column in columns:
        if column in categorical:
            train[column] = ["A", "A", "B"]
            validation[column] = ["unseen-category", "A"]
        elif column == "Company":
            train[column] = ["C1", None, "C2"]
            validation[column] = [None, "C3"]
        elif column == "LeadTime":
            train[column] = [0.0, 1.0, 2.0]
            validation[column] = [100.0, 3.0]
        elif column == "Adults":
            train[column] = [1.0, 3.0, 5.0]
            validation[column] = [100.0, 1.0]
        elif column == "Children":
            train[column] = [0.0, np.nan, 2.0]
            validation[column] = [1.0, np.nan]
        else:
            train[column] = [1.0, 2.0, 3.0]
            validation[column] = [4.0, 5.0]
    return train, validation


def test_pipeline_fits_train_and_transforms_unseen_validation_category() -> None:
    """Validation categories unseen during fit do not cause transform errors."""
    train, validation = make_data()
    preprocessor = build_preprocessor(
        "classification",
        {"classification_use_adr": False, "log_transform_leadtime": True},
    )

    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.shape[1] == transformed_validation.shape[1]
    assert not np.isnan(transformed_train).any()
    assert not np.isnan(transformed_validation).any()
    assert len(preprocessor.get_feature_names_out()) == transformed_train.shape[1]


def test_numeric_imputer_is_fitted_using_train_median_only() -> None:
    """The numeric imputer does not learn from validation values."""
    train, validation = make_data()
    train.loc[1, "Adults"] = np.nan
    validation.loc[:, "Adults"] = 999.0
    preprocessor = build_preprocessor("classification", {})
    preprocessor.fit(train)

    numeric_pipeline = preprocessor.named_steps["columns"].named_transformers_[
        "numeric"
    ]
    adults_position = list(numeric_pipeline.feature_names_in_).index("Adults")
    assert numeric_pipeline.named_steps["imputer"].statistics_[adults_position] == 3.0


def test_engineered_total_features_are_correct() -> None:
    """The stateless stage computes total nights and guests with null Children as zero."""
    train, _ = make_data()
    train.loc[0, "StaysInWeekendNights"] = 2
    train.loc[0, "StaysInWeekNights"] = 3
    train.loc[0, "Adults"] = 2
    train.loc[0, "Children"] = np.nan
    train.loc[0, "Babies"] = 1
    preprocessor = build_preprocessor("classification", {})
    engineer = preprocessor.named_steps["feature_engineering"]
    engineered = engineer.fit_transform(train)

    assert engineered.loc[0, "total_nights"] == 5
    assert engineered.loc[0, "total_guests"] == 3
    assert engineered.loc[0, "HasCompany"] == int(pd.notna(train.loc[0, "Company"]))
    assert engineered.loc[0, "HasAgent"] == int(pd.notna(train.loc[0, "Agent"]))
