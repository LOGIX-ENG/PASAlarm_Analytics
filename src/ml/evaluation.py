import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


def evaluate_clusters(
    scaled_features,
    labels,
) -> dict:

    labels = np.asarray(labels)

    non_noise = labels != -1

    cluster_labels = labels[non_noise]

    unique_clusters = set(
        cluster_labels
    )

    results = {
        "cluster_count": len(unique_clusters),
        "noise_count": int(
            (labels == -1).sum()
        ),
        "noise_percentage": float(
            (labels == -1).mean() * 100
        ),
        "silhouette_score": None,
        "davies_bouldin_score": None,
    }

    if (
        len(unique_clusters) >= 2
        and non_noise.sum() > len(unique_clusters)
    ):

        results["silhouette_score"] = (
            silhouette_score(
                scaled_features[non_noise],
                cluster_labels,
            )
        )

        results["davies_bouldin_score"] = (
            davies_bouldin_score(
                scaled_features[non_noise],
                cluster_labels,
            )
        )

    return results