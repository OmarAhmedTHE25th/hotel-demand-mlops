"""Milestone 1 baseline clustering workflow for Task C."""

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.clustering.evaluate import (
    build_cluster_profiles,
    compute_clustering_metrics,
    evaluate_k_range,
    plot_k_diagnostics,
    plot_pca_clusters,
)
from src.clustering.models import DEFAULT_CLUSTER_COUNT, get_baseline_models
from src.data.loader import load_config
from src.features.feature_sets import get_features
from src.preprocessing.pipeline import build_preprocessor


SILHOUETTE_SAMPLE_SIZE = 10_000
K_DIAGNOSTIC_RANGE = range(2, 11)


def train_baseline_clustering(
    algorithm: str = "kmeans",
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
    config: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """Fit one untuned clustering algorithm on the cleaned booking population.

    The workflow is genuinely unsupervised: no supervised target (neither
    ``IsCanceled`` nor ``ADR``) is used to fit preprocessing, fit the model, or
    choose the cluster count. The final supervised test set is never scored.
    """
    if config is None:
        config = load_config()

    models = get_baseline_models(
        random_seed=int(config.get("random_seed", 42)),
        n_clusters=n_clusters,
    )
    if algorithm not in models:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}; expected one of "
            f"{', '.join(models)}."
        )

    cleaned_path = Path(config["cleaned_data_path"])
    if not cleaned_path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{cleaned_path}'. Run 'main.py clean' first."
        )
    data = pd.read_csv(cleaned_path)

    features = get_features("clustering", config)
    leaked = sorted({"IsCanceled", "ADR"}.intersection(features))
    if leaked:
        raise ValueError(f"Supervised outcome columns must not be clustering inputs: {leaked}")
    X_raw = data.loc[:, features].copy()

    preprocessor = build_preprocessor("clustering", config, scale=True)
    model = models[algorithm]

    started = perf_counter()
    transformed = preprocessor.fit_transform(X_raw)
    labels = model.fit_predict(transformed)
    fit_seconds = perf_counter() - started

    metrics, sizes = compute_clustering_metrics(
        transformed,
        labels,
        inertia=float(model.inertia_),
        sample_size=SILHOUETTE_SAMPLE_SIZE,
        random_seed=int(config.get("random_seed", 42)),
    )
    metrics = {
        "algorithm": algorithm,
        "configuration": "untuned_baseline",
        **metrics,
        "fit_seconds": fit_seconds,
    }
    metrics_df = pd.DataFrame([metrics])

    profiles = build_cluster_profiles(X_raw, labels, adr=data["ADR"])
    diagnostics = evaluate_k_range(
        transformed,
        k_values=K_DIAGNOSTIC_RANGE,
        random_seed=int(config.get("random_seed", 42)),
        sample_size=SILHOUETTE_SAMPLE_SIZE,
    )

    output_dir = Path(config.get("outputs_dir", "outputs")) / "clustering"
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    sizes.to_frame().to_csv(output_dir / "cluster_sizes.csv")
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    diagnostics.to_csv(output_dir / "k_diagnostics.csv", index=False)
    plot_pca_clusters(
        transformed,
        labels,
        output_path=output_dir / "pca_clusters.png",
        random_seed=int(config.get("random_seed", 42)),
    )
    plot_k_diagnostics(diagnostics, output_dir / "k_diagnostics.png")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    if save_artifacts:
        joblib.dump(
            pipeline,
            artifacts_dir / f"clustering_{algorithm}_baseline.joblib",
        )

    metadata = {
        "task": "clustering",
        "algorithm": algorithm,
        "configuration": "untuned_baseline",
        "evaluation_population": "all cleaned rows (unsupervised)",
        "supervised_targets_used": False,
        "final_test_set_scored": False,
        "n_clusters": int(n_clusters),
        "n_samples": int(len(transformed)),
        "n_transformed_features": int(transformed.shape[1]),
        "features": features,
        "excluded_from_clustering": ["ADR", "IsCanceled"],
        "silhouette_sample_size": min(SILHOUETTE_SAMPLE_SIZE, len(transformed)),
        "k_diagnostic_range": list(K_DIAGNOSTIC_RANGE),
        "fit_seconds": fit_seconds,
        "config": config,
        "cleaned_data_sha256": hashlib.sha256(cleaned_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\n--- Untuned K-Means Clustering Metrics ---")
    print(metrics_df.to_string(index=False))
    print("\nCluster sizes:")
    print(sizes.to_string())
    print("\nCluster profiles:")
    print(profiles.to_string(index=False))
    print(f"\nResults written to: {output_dir}")
    if save_artifacts:
        print(f"Fitted pipeline saved to: {artifacts_dir / f'clustering_{algorithm}_baseline.joblib'}")
    print("No supervised target was used to fit or select this clustering model.")
    return metrics_df
