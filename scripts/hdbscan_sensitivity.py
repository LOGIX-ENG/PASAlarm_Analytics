from pathlib import Path
import sys

# Add the project root to Python's import path.
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


WINDOWS = [
    "5min",
    "10min",
    "15min",
    "30min",
]

MIN_CLUSTER_SIZES = [
    5,
    10,
    20,
]


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "hdbscan_sensitivity_results.csv"
)


def prepare_data():

    print("Loading alarm data...")

    df = load_alarm_data()

    print(f"Loaded {len(df):,} alarm records.")

    df = add_temporal_features(df)

    df = add_alarm_features(df)

    return df


def run_analysis(df):

    results = []

    for window in WINDOWS:

        print()
        print("=" * 70)
        print(f"WINDOW: {window}")
        print("=" * 70)

        window_df = create_alarm_windows(
            df,
            window=window,
        )

        print(
            f"Window observations: "
            f"{len(window_df):,}"
        )

        if window_df.empty:
            print("No observations. Skipping.")
            continue

        for min_cluster_size in MIN_CLUSTER_SIZES:

            print(
                f"Testing min_cluster_size="
                f"{min_cluster_size}"
            )

            if len(window_df) < min_cluster_size:
                print(
                    "Not enough observations. Skipping."
                )
                continue

            try:

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
                    min_cluster_size=min_cluster_size,
                )

                evaluation = evaluate_clusters(
                    scaled_features,
                    clusterer.labels_,
                )

                results.append(
                    {
                        "window": window,
                        "min_cluster_size":
                            min_cluster_size,
                        "observations":
                            evaluation[
                                "observation_count"
                            ],
                        "clusters":
                            evaluation[
                                "cluster_count"
                            ],
                        "noise_count":
                            evaluation[
                                "noise_count"
                            ],
                        "noise_percentage":
                            evaluation[
                                "noise_percentage"
                            ],
                        "silhouette_score":
                            evaluation[
                                "silhouette_score"
                            ],
                        "davies_bouldin_score":
                            evaluation[
                                "davies_bouldin_score"
                            ],
                        "features":
                            ", ".join(feature_names),
                    }
                )

                print(
                    f"  Clusters: "
                    f"{evaluation['cluster_count']}"
                )

                print(
                    f"  Noise: "
                    f"{evaluation['noise_percentage']:.2f}%"
                )

                print(
                    f"  Silhouette: "
                    f"{evaluation['silhouette_score']}"
                )

                print(
                    f"  Davies-Bouldin: "
                    f"{evaluation['davies_bouldin_score']}"
                )

            except Exception as exc:

                print(
                    f"  ERROR: {exc}"
                )

    return pd.DataFrame(results)


def main():

    df = prepare_data()

    results = run_analysis(df)

    if results.empty:
        print()
        print(
            "No HDBSCAN results were generated."
        )
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("HDBSCAN SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(results.to_string(index=False))

    print()
    print(
        f"Results saved to:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()