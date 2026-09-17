import hdbscan
import pandas as pd


def run_hdbscan(
    scaled_features,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
):
    """Run HDBSCAN clustering."""

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    clusterer.fit(scaled_features)

    return clusterer


def add_cluster_results(
    window_df: pd.DataFrame,
    clusterer,
) -> pd.DataFrame:

    result = window_df.copy()

    result["Cluster"] = (
        clusterer.labels_
    )

    result["ClusterProbability"] = (
        clusterer.probabilities_
    )

    result["IsNoise"] = (
        result["Cluster"] == -1
    )

    return result