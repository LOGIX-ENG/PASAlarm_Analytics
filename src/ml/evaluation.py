import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


def evaluate_clusters(
    scaled_features,
    labels,
) -> dict:
    """
    Evaluate HDBSCAN clustering.

    Noise observations (-1) are excluded from the
    silhouette and Davies-Bouldin calculations.

    Noise percentage is reported separately.
    """

    labels = np.asarray(labels)

    total_observations = len(labels)

    noise_mask = labels == -1
    non_noise_mask = ~noise_mask

    noise_count = int(
        noise_mask.sum()
    )

    noise_percentage = (
        noise_count / total_observations * 100
        if total_observations > 0
        else 0.0
    )

    non_noise_labels = labels[
        non_noise_mask
    ]

    unique_clusters = np.unique(
        non_noise_labels
    )

    results: dict[str, int | float | None] = {
        "observation_count": total_observations,
        "cluster_count": len(unique_clusters),
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "silhouette_score": None,
        "davies_bouldin_score": None,
    }

    # At least two clusters are required for both
    # clustering quality metrics.
    if (
        len(unique_clusters) >= 2
        and len(non_noise_labels) > len(unique_clusters)
    ):

        non_noise_features = (
            scaled_features[non_noise_mask]
        )

        results["silhouette_score"] = (
            silhouette_score(
                non_noise_features,
                non_noise_labels,
            )
        )

        results["davies_bouldin_score"] = (
            davies_bouldin_score(
                non_noise_features,
                non_noise_labels,
            )
        )

    return results