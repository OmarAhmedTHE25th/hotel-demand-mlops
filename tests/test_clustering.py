"""Tests for the untuned Task C clustering workflow."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.clustering.evaluate import (
    build_cluster_profiles,
    compute_clustering_metrics,
    evaluate_k_range,
    plot_pca_clusters,
)
from src.clustering.models import get_baseline_models
from src.features.feature_sets import get_features
from src.preprocessing.pipeline import build_preprocessor


SEGMENTS = [
    {
        "LeadTime": 5.0,
        "StaysInWeekendNights": 0.0,
        "StaysInWeekNights": 1.0,
        "Adults": 1.0,
        "Children": 0.0,
        "Babies": 0.0,
        "IsRepeatedGuest": 0.0,
        "TotalOfSpecialRequests": 0.0,
        "MarketSegment": "Online TA",
        "DistributionChannel": "TA/TO",
        "DepositType": "No Deposit",
        "CustomerType": "Transient",
    },
    {
        "LeadTime": 120.0,
        "StaysInWeekendNights": 2.0,
        "StaysInWeekNights": 3.0,
        "Adults": 2.0,
        "Children": 1.0,
        "Babies": 0.0,
        "IsRepeatedGuest": 0.0,
        "TotalOfSpecialRequests": 2.0,
        "MarketSegment": "Groups",
        "DistributionChannel": "Direct",
        "DepositType": "Non Refund",
        "CustomerType": "Transient-Party",
    },
    {
        "LeadTime": 30.0,
        "StaysInWeekendNights": 1.0,
        "StaysInWeekNights": 1.0,
        "Adults": 2.0,
        "Children": 0.0,
        "Babies": 0.0,
        "IsRepeatedGuest": 1.0,
        "TotalOfSpecialRequests": 1.0,
        "MarketSegment": "Corporate",
        "DistributionChannel": "Corporate",
        "DepositType": "No Deposit",
        "CustomerType": "Contract",
    },
    {
        "LeadTime": 250.0,
        "StaysInWeekendNights": 3.0,
        "StaysInWeekNights": 4.0,
        "Adults": 2.0,
        "Children": 0.0,
        "Babies": 1.0,
        "IsRepeatedGuest": 1.0,
        "TotalOfSpecialRequests": 3.0,
        "MarketSegment": "Offline TA/TO",
        "DistributionChannel": "TA/TO",
        "DepositType": "Refundable",
        "CustomerType": "Transient",
    },
]


def make_clustering_data(rows_per_segment: int = 50) -> pd.DataFrame:
    """Create separable synthetic booking segments without using H2.csv."""
    rng = np.random.default_rng(7)
    features = get_features("clustering", {})
    numeric_columns = [
        column
        for column in features
        if column
        not in {
            "MarketSegment",
            "DistributionChannel",
            "DepositType",
            "CustomerType",
        }
    ]

    rows = []
    for segment in SEGMENTS:
        for _ in range(rows_per_segment):
            row = {
                column: segment[column]
                + rng.normal(0.0, 1.0)
                for column in numeric_columns
            }
            for column in (
                "MarketSegment",
                "DistributionChannel",
                "DepositType",
                "CustomerType",
            ):
                row[column] = segment[column]
            rows.append(row)
    return pd.DataFrame(rows).loc[:, features]


def fit_baseline(seed: int = 42, n_clusters: int = 4):
    """Fit the shared preprocessor and return transformed data plus labels."""
    data = make_clustering_data()
    preprocessor = build_preprocessor("clustering", {"log_transform_leadtime": True})
    transformed = preprocessor.fit_transform(data)
    model = get_baseline_models(random_seed=seed, n_clusters=n_clusters)["kmeans"]
    labels = model.fit_predict(transformed)
    return data, preprocessor, transformed, model, labels


def test_clustering_features_exclude_supervised_outcomes() -> None:
    """Neither the classification nor the regression target is a clustering input."""
    features = get_features("clustering", {})
    assert "IsCanceled" not in features
    assert "ADR" not in features
    assert "LeadTime" in features


def test_preprocessing_and_kmeans_labels_are_deterministic() -> None:
    """The seeded baseline assigns identical labels on repeated runs."""
    data, _, transformed, _, labels = fit_baseline()
    _, _, second_transformed, second_model, second_labels = fit_baseline()

    assert not np.isnan(np.asarray(transformed)).any()
    np.testing.assert_allclose(transformed, second_transformed)
    assert len(labels) == len(data)
    assert set(labels) == set(range(4))
    assert np.array_equal(labels, second_labels)


def test_internal_metrics_and_cluster_sizes() -> None:
    """Internal metrics are finite and sizes cover the whole population."""
    data, _, transformed, model, labels = fit_baseline()
    metrics, sizes = compute_clustering_metrics(
        transformed,
        labels,
        inertia=model.inertia_,
        sample_size=100,
        random_seed=42,
    )

    assert metrics["n_samples"] == len(data)
    assert metrics["n_clusters"] == 4
    assert -1.0 <= metrics["silhouette"] <= 1.0
    assert metrics["davies_bouldin"] >= 0.0
    assert metrics["calinski_harabasz"] > 0.0
    assert int(sizes.sum()) == len(data)
    assert list(sizes.index) == [0, 1, 2, 3]


def test_cluster_profiles_use_booking_attributes_and_post_hoc_adr() -> None:
    """Profiles summarize every cluster without using ADR during fitting."""
    data, _, _, _, labels = fit_baseline()
    adr = pd.Series(np.linspace(50.0, 250.0, len(data)))
    profiles = build_cluster_profiles(data, labels, adr=adr)

    assert len(profiles) == 4
    assert int(profiles["n_bookings"].sum()) == len(data)
    assert np.isclose(profiles["share"].sum(), 1.0)
    assert "post_hoc_mean_ADR" in profiles.columns
    assert "MarketSegment_top" in profiles.columns
    assert "MarketSegment_top_share" in profiles.columns


def test_pca_plot_and_pipeline_artifact_round_trip(tmp_path: Path) -> None:
    """A saved pipeline reproduces its cluster assignments after loading."""
    data, preprocessor, transformed, model, labels = fit_baseline()
    plot_path = tmp_path / "pca_clusters.png"
    plot_pca_clusters(transformed, labels, plot_path)
    assert plot_path.is_file() and plot_path.stat().st_size > 0

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    artifact_path = tmp_path / "clustering_kmeans_baseline.joblib"
    joblib.dump(pipeline, artifact_path)
    loaded = joblib.load(artifact_path)

    assert np.array_equal(loaded.predict(data), labels)


def test_k_diagnostics_report_seeded_cluster_counts() -> None:
    """The descriptive sweep returns one row per requested k."""
    _, _, transformed, _, _ = fit_baseline()
    diagnostics = evaluate_k_range(transformed, k_values=(2, 3), sample_size=100)

    assert list(diagnostics["k"]) == [2, 3]
    assert diagnostics["silhouette"].between(-1.0, 1.0).all()
    assert (diagnostics["davies_bouldin"] >= 0.0).all()
