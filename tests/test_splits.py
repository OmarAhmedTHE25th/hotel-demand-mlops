"""Tests for reproducible task-specific data splits."""

from typing import Any

import pandas as pd

from src.features.feature_sets import (
    CLASSIFICATION_FEATURES,
    CLUSTERING_FEATURES,
    REGRESSION_FEATURES,
    get_features,
)
from src.preprocessing.splits import split_classification, split_regression


def make_synthetic_data(row_count: int = 400) -> pd.DataFrame:
    """Create all project columns without requiring H2.csv."""
    data: dict[str, Any] = {}
    for column in set(
        CLASSIFICATION_FEATURES
        + REGRESSION_FEATURES
        + CLUSTERING_FEATURES
        + [
            "IsCanceled",
            "ReservationStatus",
            "ReservationStatusDate",
            "BookingChanges",
            "AssignedRoomType",
            "DaysInWaitingList",
        ]
    ):
        if column in {"ArrivalDateMonth", "Meal", "Country", "MarketSegment",
                      "DistributionChannel", "ReservedRoomType", "DepositType",
                      "Agent", "Company", "CustomerType", "AssignedRoomType",
                      "ReservationStatus"}:
            data[column] = ["A", "B", "C", "D"] * (row_count // 4) + ["A"] * (row_count % 4)
        elif column == "ReservationStatusDate":
            data[column] = ["2017-01-01"] * row_count
        elif column == "IsCanceled":
            data[column] = [index % 2 for index in range(row_count)]
        elif column == "ADR":
            data[column] = [100.0 + index % 20 for index in range(row_count)]
        else:
            data[column] = [index % 10 for index in range(row_count)]
    return pd.DataFrame(data)


def split_config(tmp_path: Any) -> dict[str, Any]:
    """Return deterministic split settings and an isolated indices path."""
    return {
        "random_seed": 42,
        "val_size": 0.15,
        "test_size": 0.15,
        "adr_max": 500,
        "classification_use_adr": False,
        "splits_dir": str(tmp_path),
    }


def assert_disjoint_indices(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Assert the returned subsets contain distinct source rows."""
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)


def test_classification_split_sizes_stratification_and_leakage(tmp_path: Any) -> None:
    """Classification splitting preserves class ratios and excludes leakage."""
    data = make_synthetic_data()
    result = split_classification(data, split_config(tmp_path))
    X_train, X_val, X_test, y_train, y_val, y_test = result

    assert_disjoint_indices(X_train, X_val, X_test)
    assert (len(X_train), len(X_val), len(X_test)) == (280, 60, 60)
    overall_rate = data["IsCanceled"].mean()
    assert abs(y_train.mean() - overall_rate) < 0.02
    assert abs(y_val.mean() - overall_rate) < 0.02
    assert abs(y_test.mean() - overall_rate) < 0.02
    for X in (X_train, X_val, X_test):
        assert not {
            "ReservationStatus",
            "ReservationStatusDate",
            "BookingChanges",
            "AssignedRoomType",
            "DaysInWaitingList",
            "IsCanceled",
        }.intersection(X.columns)
        assert "ADR" not in X.columns


def test_regression_split_filters_adr_and_excludes_leakage(tmp_path: Any) -> None:
    """Regression splitting excludes target/leakage columns and applies ADR filter."""
    data = make_synthetic_data()
    data.loc[0, "ADR"] = 0.0
    data.loc[1, "ADR"] = 501.0
    X_train, X_val, X_test, y_train, y_val, y_test = split_regression(
        data, split_config(tmp_path)
    )

    assert_disjoint_indices(X_train, X_val, X_test)
    assert (len(X_train), len(X_val), len(X_test)) == (278, 60, 60)
    for X in (X_train, X_val, X_test):
        assert not {
            "ReservationStatus",
            "ReservationStatusDate",
            "BookingChanges",
            "AssignedRoomType",
            "DaysInWaitingList",
            "IsCanceled",
            "ADR",
        }.intersection(X.columns)
    assert all(y_split.gt(0).all() and y_split.le(500).all()
               for y_split in (y_train, y_val, y_test))


def test_same_seed_produces_same_split(tmp_path: Any) -> None:
    """Repeated splitting with identical settings yields the same row indices."""
    data = make_synthetic_data()
    config = split_config(tmp_path)
    first = split_classification(data, config)
    second = split_classification(data, config)

    for first_split, second_split in zip(first, second):
        assert first_split.index.tolist() == second_split.index.tolist()
        assert first_split.equals(second_split)


def test_classification_adr_is_configurable() -> None:
    """ADR is excluded by default and included only when explicitly enabled."""
    assert "ADR" not in get_features("classification", {})
    assert "ADR" in get_features("classification", {"classification_use_adr": True})
