"""Reusable task-specific preprocessing pipelines."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.feature_sets import get_features


LOW_CARDINALITY_CATEGORICALS = [
    "ArrivalDateMonth",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
]
HIGH_CARDINALITY_CATEGORICALS = ["Country", "Agent"]


def _categoricals_to_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize category values to strings while retaining missing values."""
    string_values = X.astype("string").astype(object)
    return string_values.where(X.notna(), np.nan)


class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stateless booking features while retaining original column names."""

    def __init__(
        self,
        feature_columns: list[str],
        log_transform_leadtime: bool = True,
    ) -> None:
        self.feature_columns = feature_columns
        self.log_transform_leadtime = log_transform_leadtime

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BookingFeatureEngineer":
        """Store input and output column names; no statistics are learned."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        output_columns = [column for column in self.feature_columns if column != "Company"]
        if {
            "StaysInWeekendNights",
            "StaysInWeekNights",
        }.issubset(self.feature_columns):
            output_columns.append("total_nights")
        if {"Adults", "Children", "Babies"}.issubset(self.feature_columns):
            output_columns.append("total_guests")
        if "Company" in self.feature_columns:
            output_columns.append("HasCompany")
        if "Agent" in self.feature_columns:
            output_columns.append("HasAgent")
        self.feature_names_out_ = np.asarray(output_columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply engineered columns without fitting or modifying the input."""
        transformed = X.loc[:, self.feature_columns].copy()
        if self.log_transform_leadtime and "LeadTime" in transformed.columns:
            transformed["LeadTime"] = np.log1p(transformed["LeadTime"])

        if {
            "StaysInWeekendNights",
            "StaysInWeekNights",
        }.issubset(self.feature_columns):
            transformed["total_nights"] = (
                transformed["StaysInWeekendNights"]
                + transformed["StaysInWeekNights"]
            )
        if {"Adults", "Children", "Babies"}.issubset(self.feature_columns):
            transformed["total_guests"] = (
                transformed["Adults"]
                + transformed["Children"].fillna(0)
                + transformed["Babies"]
            )
        if "Company" in transformed.columns:
            transformed["HasCompany"] = transformed["Company"].notna().astype("int64")
            transformed = transformed.drop(columns="Company")
        if "Agent" in transformed.columns:
            transformed["HasAgent"] = transformed["Agent"].notna().astype("int64")
        return transformed

    def get_feature_names_out(
        self, input_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return names after stateless feature engineering."""
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("BookingFeatureEngineer must be fitted before requesting names.")
        return self.feature_names_out_.copy()


def build_preprocessor(
    task: str, config: dict[str, Any], scale: bool = True
) -> Pipeline:
    """Build preprocessing for a task; fit only on training data.

    Feature engineering and LeadTime log transformation are stateless. Median
    and most-frequent imputers, one-hot encoder categories/frequencies, and
    StandardScaler statistics learn from data. Call ``fit`` on training data
    only, then use ``transform`` for validation and test data.
    """
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError(
            "task must be 'classification', 'regression', or 'clustering'."
        )

    feature_columns = get_features(task, config)
    engineered_columns = list(feature_columns)
    if {"StaysInWeekendNights", "StaysInWeekNights"}.issubset(feature_columns):
        engineered_columns.append("total_nights")
    if {"Adults", "Children", "Babies"}.issubset(feature_columns):
        engineered_columns.append("total_guests")
    if "Company" in feature_columns:
        engineered_columns.append("HasCompany")
        engineered_columns.remove("Company")
    if "Agent" in feature_columns:
        engineered_columns.append("HasAgent")

    high_cardinality = [
        column
        for column in HIGH_CARDINALITY_CATEGORICALS
        if column in feature_columns
    ]
    low_cardinality = [
        column
        for column in LOW_CARDINALITY_CATEGORICALS
        if column in feature_columns
    ]
    categorical_columns = set(high_cardinality + low_cardinality)
    numeric_columns = [
        column for column in engineered_columns if column not in categorical_columns
    ]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if low_cardinality:
        transformers.append(
            (
                "low_cardinality",
                Pipeline(
                    [
                        (
                            "stringify",
                            FunctionTransformer(
                                _categoricals_to_strings,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                low_cardinality,
            )
        )
    if high_cardinality:
        transformers.append(
            (
                "high_cardinality",
                Pipeline(
                    [
                        (
                            "stringify",
                            FunctionTransformer(
                                _categoricals_to_strings,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="Unknown"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=0.01,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                high_cardinality,
            )
        )

    return Pipeline(
        [
            (
                "feature_engineering",
                BookingFeatureEngineer(
                    feature_columns,
                    log_transform_leadtime=bool(
                        config.get("log_transform_leadtime", True)
                    ),
                ),
            ),
            (
                "columns",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
