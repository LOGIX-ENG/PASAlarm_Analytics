from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.load_data import load_alarm_data
from src.features.temporal_features import add_temporal_features
from src.features.alarm_features import add_alarm_features
from src.features.window_features import create_alarm_windows
from src.ml.preprocessing import prepare_ml_data
from src.ml.hdbscan_model import run_hdbscan
from src.ml.evaluation import evaluate_clusters


WINDOW = "5min"
MIN_CLUSTER_SIZE = 10


def main():

    print("Loading alarm data...")

    df = load_alarm_data()

    print(f"Alarm records: {len(df):,}")

    df = add_temporal_features(df)
    df = add_alarm_features(df)

    window_df = create_alarm_windows(
        df,
        window=WINDOW,
    )

    print(
        f"Window observations: "
        f"{len(window_df):,}"
    )

    (
        scaled_features,
        scaler,
        feature_names,
        feature_data,
    ) = prepare_ml_data(
        window_df
    )

    clusterer = run_hdbscan(
        scaled_features,
        min_cluster_size=MIN_CLUSTER_SIZE,
    )

    result = window_df.copy()

    result["Cluster"] = clusterer.labels_
    result["ClusterProbability"] = (
        clusterer.probabilities_
    )

    result["IsNoise"] = (
        result["Cluster"] == -1
    )

    evaluation = evaluate_clusters(
        scaled_features,
        clusterer.labels_,
    )

    print()
    print("=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(f"Window: {WINDOW}")
    print(
        f"Minimum cluster size: "
        f"{MIN_CLUSTER_SIZE}"
    )

    print(
        f"Observations: "
        f"{evaluation['observation_count']:,}"
    )

    print(
        f"Clusters: "
        f"{evaluation['cluster_count']}"
    )

    print(
        f"Noise: "
        f"{evaluation['noise_count']:,}"
    )

    print(
        f"Noise percentage: "
        f"{evaluation['noise_percentage']:.2f}%"
    )

    print(
        f"Silhouette: "
        f"{evaluation['silhouette_score']:.6f}"
    )

    print(
        f"Davies-Bouldin: "
        f"{evaluation['davies_bouldin_score']:.6f}"
    )

    print()
    print("=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)

    cluster_summary = (
        result[
            result["Cluster"] != -1
        ]
        .groupby("Cluster")
        .agg(
            Observations=("Cluster", "size"),
            AvgAlarmCount=(
                "AlarmCount",
                "mean",
            ),
            AvgUniqueTags=(
                "UniqueTags",
                "mean",
            ),
            AvgPriority=(
                "AveragePriority",
                "mean",
            ),
            AvgActive=(
                "ActiveCount",
                "mean",
            ),
            AvgAcked=(
                "AckedCount",
                "mean",
            ),
            AvgNormal=(
                "NormalCount",
                "mean",
            ),
            AvgHighPriority=(
                "HighPriorityCount",
                "mean",
            ),
            AvgAlarmRate=(
                "AlarmRatePerMinute",
                "mean",
            ),
            AvgProbability=(
                "ClusterProbability",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Observations",
            ascending=False,
        )
    )

    print(
        cluster_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("NOISE SUMMARY")
    print("=" * 70)

    noise = result[
        result["Cluster"] == -1
    ]

    if not noise.empty:

        print(
            noise[
                [
                    "Window",
                    "Area",
                    "Field",
                    "AlarmCount",
                    "UniqueTags",
                    "AveragePriority",
                    "MaximumPriority",
                    "ActiveCount",
                    "AckedCount",
                    "NormalCount",
                    "HighPriorityCount",
                    "AlarmRatePerMinute",
                    "ClusterProbability",
                ]
            ]
            .sort_values(
                "AlarmCount",
                ascending=False,
            )
            .head(25)
            .to_string(index=False)
        )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "selected_hdbscan_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()