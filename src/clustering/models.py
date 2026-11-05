"""Model definitions and factories for Task C clustering."""

from typing import Any, Dict

from sklearn.cluster import KMeans


DEFAULT_CLUSTER_COUNT = 5


def get_baseline_models(
    random_seed: int = 42,
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
) -> Dict[str, Any]:
    """Return the untuned baseline clustering algorithms.

    K-Means uses a fixed, pre-declared cluster count. The baseline is not
    selected with supervised labels; systematic parameter search belongs to
    the tuning milestone.
    """
    return {
        "kmeans": KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_seed,
        ),
    }
