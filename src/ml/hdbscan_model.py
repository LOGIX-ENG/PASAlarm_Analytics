import hdbscan
import pandas as pd


def run_hdbscan(
    scaled_features,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
):
    """
    Run HDBSCAN on standardized alarm-window features.

    HDBSCAN label -1 represents observations classified as noise.
    """

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    if len(scaled_features) < min_cluster_size:
        raise ValueError(
            "The number of observations must be greater than "
            "or equal to min_cluster_size."
        )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    clusterer.fit(
        scaled_features
    )

    return clusterer


def add_cluster_results(
    window_df: pd.DataFrame,
    clusterer,
) -> pd.DataFrame:
    """
    Add HDBSCAN labels and membership probabilities
    to the window-level observations.
    """

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