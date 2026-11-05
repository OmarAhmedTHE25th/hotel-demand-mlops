"""Unsupervised evaluation utilities for booking clusters."""

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_clustering_metrics(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Sequence[int],
    inertia: Optional[float] = None,
    sample_size: int = 10_000,
    random_seed: int = 42,
) -> tuple[dict[str, float], pd.Series]:
    """Compute internal metrics and cluster sizes for a fitted labelling.

    Silhouette is estimated on a seeded random sample because its pair-wise
    distance computation is quadratic in the number of rows.
    """
    labels_array = np.asarray(labels)
    n_samples = int(len(labels_array))
    n_clusters = int(np.unique(labels_array).size)
    if n_clusters < 2:
        raise ValueError("At least two clusters are required for internal metrics.")
    if n_clusters >= n_samples:
        raise ValueError("Cluster count must be smaller than the number of samples.")

    metrics = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "inertia": float(inertia) if inertia is not None else np.nan,
        "silhouette": float(
            silhouette_score(
                X,
                labels_array,
                sample_size=min(sample_size, n_samples),
                random_state=random_seed,
            )
        ),
        "davies_bouldin": float(davies_bouldin_score(X, labels_array)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels_array)),
    }
    sizes = (
        pd.Series(labels_array, name="cluster")
        .value_counts()
        .sort_index()
        .rename("n_bookings")
        .rename_axis("cluster")
    )
    return metrics, sizes


def evaluate_k_range(
    X: Union[np.ndarray, pd.DataFrame],
    k_values: Sequence[int],
    random_seed: int = 42,
    n_init: int = 10,
    sample_size: int = 10_000,
) -> pd.DataFrame:
    """Report diagnostics for several cluster counts.

    This is a descriptive sweep to inform later tuning. It does not change the
    untuned baseline configuration used for the reported model.
    """
    rows = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            random_state=random_seed,
        )
        labels = model.fit_predict(X)
        scores, _ = compute_clustering_metrics(
            X,
            labels,
            inertia=model.inertia_,
            sample_size=sample_size,
            random_seed=random_seed,
        )
        rows.append({"k": int(k), **scores})
    return pd.DataFrame(rows)


def build_cluster_profiles(
    X: pd.DataFrame,
    labels: Sequence[int],
    adr: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Summarize clusters with their original booking attributes.

    Numeric attributes are summarized by their cluster mean; categorical
    attributes by their most frequent level and that level's share. ``adr`` is
    reported post hoc only: it is not a clustering input and is not used to fit
    the model or choose the number of clusters.
    """
    frame = X.copy()
    if adr is not None:
        frame["_post_hoc_ADR"] = np.asarray(adr, dtype=float)
    frame["_cluster"] = np.asarray(labels)

    numeric_columns = [
        column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]

    rows = []
    for cluster_id, group in frame.groupby("_cluster", sort=True):
        row: dict[str, object] = {
            "cluster": int(cluster_id),
            "n_bookings": int(len(group)),
            "share": float(len(group) / len(frame)),
        }
        for column in numeric_columns:
            row[column] = float(group[column].mean())
        for column in categorical_columns:
            counts = group[column].astype("string").value_counts(dropna=False)
            row[f"{column}_top"] = str(counts.index[0])
            row[f"{column}_top_share"] = float(counts.iloc[0] / len(group))
        if adr is not None:
            row["post_hoc_mean_ADR"] = float(group["_post_hoc_ADR"].mean())
        rows.append(row)

    profiles = pd.DataFrame(rows)
    ordered = ["cluster", "n_bookings", "share"] + numeric_columns
    for column in categorical_columns:
        ordered.extend([f"{column}_top", f"{column}_top_share"])
    if adr is not None:
        ordered.append("post_hoc_mean_ADR")
    return profiles.loc[:, ordered]


def plot_pca_clusters(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Sequence[int],
    output_path: Union[str, Path],
    random_seed: int = 42,
) -> None:
    """Save a two-dimensional PCA projection coloured by cluster."""
    labels_array = np.asarray(labels)
    projection = PCA(n_components=2, random_state=random_seed)
    coordinates = projection.fit_transform(np.asarray(X))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id in np.unique(labels_array):
        mask = labels_array == cluster_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.6,
            label=f"Cluster {cluster_id}",
        )
    ax.set(
        title="K-Means clusters in a two-dimensional PCA projection",
        xlabel=f"PC1 ({projection.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({projection.explained_variance_ratio_[1]:.1%} variance)",
    )
    ax.legend(markerscale=2, title="Cluster")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_k_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: Union[str, Path],
) -> None:
    """Save inertia, silhouette, and Davies-Bouldin curves across k."""
    panels = (
        ("inertia", "Inertia (lower is tighter)"),
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for ax, (column, title) in zip(axes, panels):
        ax.plot(diagnostics["k"], diagnostics[column], marker="o")
        ax.set(xlabel="Number of clusters (k)", ylabel=column, title=title)
        ax.grid(alpha=0.3)
    fig.suptitle("Cluster-count diagnostics (not used to fit the baseline)")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
