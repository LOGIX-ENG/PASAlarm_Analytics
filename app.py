# *********************************************************************
# Imports
# *********************************************************************
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# *********************************************************************
# Initial Page Setup
# *********************************************************************
st.set_page_config(
    page_title="Industrial Alarm Analytics",
    page_icon="",
    layout="wide",
)

# *********************************************************************
# Load Sanatized Alarm Dataset
# *********************************************************************
@st.cache_data
def load_data():
    """Load and prepare the sanitized alarm dataset from CSV."""
    data_path = Path(__file__).parent / "data" / "alarms.csv"
    df = pd.read_csv(data_path)
    df["Time / Date"] = pd.to_datetime(df["Time / Date"], errors="coerce")
    df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce")
    df = df.dropna(subset=["Time / Date"]).copy()
    return df.sort_values("Time / Date")

# *********************************************************************
# Create Alarming events. These are separated into Five/Minute Observations
# *********************************************************************
def create_ml_windows(df):
    """Convert alarm events into five-minute Area/Field observations."""
    data = df.copy()
    data["Window"] = data["Time / Date"].dt.floor("5min")
    data["ActiveFlag"] = data["Alarm State"].astype(str).str.lower().eq("active").astype(int)
    data["AckedFlag"] = data["Alarm State"].astype(str).str.lower().eq("acked").astype(int)
    data["NormalFlag"] = data["Alarm State"].astype(str).str.lower().eq("normal").astype(int)
    data["HighPriorityFlag"] = data["Priority"].fillna(0).ge(500).astype(int)

    grouped = (
        data.groupby(["Window", "Area", "Field"], dropna=False)
        .agg(
            AlarmCount=("Tag", "count"),
            UniqueTags=("Tag", "nunique"),
            AveragePriority=("Priority", "mean"),
            MaximumPriority=("Priority", "max"),
            ActiveCount=("ActiveFlag", "sum"),
            AckedCount=("AckedFlag", "sum"),
            NormalCount=("NormalFlag", "sum"),
            HighPriorityCount=("HighPriorityFlag", "sum"),
        )
        .reset_index()
        .sort_values("Window")
    )
    grouped["AlarmRatePerMinute"] = grouped["AlarmCount"] / 5
    return grouped


FEATURE_COLUMNS = [
    "AlarmCount", "UniqueTags", "AveragePriority", "MaximumPriority",
    "ActiveCount", "AckedCount", "NormalCount", "HighPriorityCount",
    "AlarmRatePerMinute",
]

# *********************************************************************
# Extract the Column Features
# "AlarmCount", "UniqueTags", "AveragePriority", "MaximumPriority",
# "ActiveCount", "AckedCount", "NormalCount", "HighPriorityCount",
# "AlarmRatePerMinute",
#*********************************************************************
def extract_features(window_data):
    return (
        window_data[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

# *********************************************************************
# Evaluation of the Cluster using the Silhouette Score and
# the Davis Bouldin Score. These features are imported from SKLearn
# *********************************************************************
def evaluate_clustering(features, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2 or mask.sum() < 2:
        return None

    return {
        "silhouette": silhouette_score(features[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(features[mask], labels[mask]),
        "n_clusters_scored": n_clusters,
    }

# *********************************************************************
# Run the HDBSCAN clustering algorithm on the hisorical
# Five-Minue Observations.
# *********************************************************************

def run_hdbscan(window_data):
    """Run descriptive HDBSCAN on historical five-minute observations."""
    features = extract_features(window_data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = model.fit_predict(scaled_features)

    results = window_data.copy()
    results["Cluster"] = labels
    return results, scaled_features

# *********************************************************************
# All five-minute observations will be sorted chronologically.
# The earliest approximately 80 percent will form the training set and
# the latest approximately 20 percent will form the holdout set.
# *********************************************************************

def predict_holdout_membership(all_windows):
    """Train on the earliest 80% and classify the held-out 20%."""
    windows = all_windows.sort_values("Window").reset_index(drop=True)
    cut = int(len(windows) * 0.8)

    train_windows = windows.iloc[:cut].copy()
    holdout_windows = windows.iloc[cut:].copy()

    train_features = extract_features(train_windows)
    holdout_features = extract_features(holdout_windows)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    holdout_scaled = scaler.transform(holdout_features)

    model = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
    train_labels = model.fit_predict(train_scaled)
    predicted_labels, strengths = hdbscan.approximate_predict(model, holdout_scaled)

    holdout_results = holdout_windows.copy()
    holdout_results["PredictedCluster"] = predicted_labels
    holdout_results["MembershipStrength"] = strengths
    holdout_results["MatchesKnownPattern"] = predicted_labels != -1

    return {
        "train_windows": len(train_windows),
        "holdout_windows": len(holdout_windows),
        "holdout_results": holdout_results,
        "matched": int(holdout_results["MatchesKnownPattern"].sum()),
        "unmatched": int((~holdout_results["MatchesKnownPattern"]).sum()),
        "train_metrics": evaluate_clustering(train_scaled, train_labels),
    }

# *********************************************************************
# Begin Streamlit Applicaiton
# *********************************************************************

def main():
    st.title("Industrial Alarm Analytics")
    st.write(
        "Interactive analysis of historical industrial alarm activity "
        "using descriptive analytics and unsupervised machine learning."
    )

    df = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Alarm Records", f"{len(df):,}")
    col2.metric("Areas", df["Area"].nunique())
    col3.metric("Fields", df["Field"].nunique())

    st.subheader("Explore the Data")
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        area_options = ["All"] + sorted(
            df["Area"].dropna().astype(str).unique().tolist()
        )
        selected_area = st.selectbox("Area", area_options)

    with filter_col2:
        priority_options = ["All"] + sorted(
            df["Priority"].dropna().unique().tolist()
        )
        selected_priority = st.selectbox("Priority", priority_options)

    filtered_df = df.copy()

    if selected_area != "All":
        filtered_df = filtered_df[
            filtered_df["Area"].astype(str) == selected_area
        ]

    if selected_priority != "All":
        filtered_df = filtered_df[
            filtered_df["Priority"] == selected_priority
        ]

    filtered_df = filtered_df.copy()

    st.write(
        f"Showing **{len(filtered_df):,}** of **{len(df):,}** alarm records."
    )

    st.subheader("1. Alarm Events by Area")
    area_counts = filtered_df["Area"].value_counts().sort_values(ascending=False)
    st.bar_chart(area_counts)

    st.subheader("2. Alarm Events Over Time")
    daily_counts = filtered_df.set_index("Time / Date").resample("D").size()
    st.line_chart(daily_counts)

    st.subheader("Filtered Alarm Records")
    display_columns = [
        c for c in [
            "Time / Date", "Tag", "Priority", "Type", "Quality",
            "Alarm State", "Area", "Field"
        ]
        if c in filtered_df.columns
    ]
    st.dataframe(
        filtered_df[display_columns].head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("3. HDBSCAN Pattern Discovery (descriptive)")
    st.write(
        "HDBSCAN groups similar five-minute Area/Field alarm-activity "
        "observations within the current filter selection. Observations "
        "classified as noise do not fit the learned density structure."
    )

    if st.button("Run HDBSCAN", type="primary"):
        if len(filtered_df) < 20:
            st.warning("There is not enough filtered data to perform the analysis.")
        else:
            with st.spinner("Running HDBSCAN..."):
                window_data = create_ml_windows(filtered_df)

                if len(window_data) < 20:
                    st.warning(
                        "The selected data does not produce enough five-minute "
                        "observations for analysis."
                    )
                else:
                    results, scaled_features = run_hdbscan(window_data)

                    cluster_labels = set(results["Cluster"])
                    cluster_count = len(cluster_labels - {-1})
                    noise_count = int((results["Cluster"] == -1).sum())
                    noise_percent = noise_count / len(results) * 100

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Clusters", cluster_count)
                    r2.metric("Noise Observations", noise_count)
                    r3.metric("Noise %", f"{noise_percent:.2f}%")

                    metrics = evaluate_clustering(
                        scaled_features,
                        results["Cluster"],
                    )

                    if metrics:
                        m1, m2 = st.columns(2)
                        m1.metric(
                            "Silhouette Score",
                            f"{metrics['silhouette']:.4f}",
                        )
                        m2.metric(
                            "Davies-Bouldin Index",
                            f"{metrics['davies_bouldin']:.4f}",
                        )
                    else:
                        st.info(
                            "Not enough cluster structure in this selection "
                            "to compute Silhouette / Davies-Bouldin."
                        )

                    pca = PCA(n_components=2)
                    coordinates = pca.fit_transform(scaled_features)

                    plot_data = pd.DataFrame({
                        "PC1": coordinates[:, 0],
                        "PC2": coordinates[:, 1],
                        "Cluster": results["Cluster"].astype(str),
                    })

                    st.scatter_chart(
                        plot_data,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                    )

                    st.caption(
                        "Each point represents one five-minute Area/Field "
                        "observation. PCA is used only to display the HDBSCAN "
                        "results in two dimensions."
                    )

    st.subheader(
        "4. Classify Holdout Observations Against Historical Patterns"
    )
    st.write(
        "HDBSCAN is trained on the earliest approximately 80% of historical "
        "five-minute Area/Field observations. The remaining approximately 20% "
        "are held out from model fitting and classified against the learned "
        "clusters using approximate_predict(). An observation can be assigned "
        "to an established cluster or classified as noise when it does not "
        "match the learned density structure."
    )

    if st.button("Train Model & Classify Holdout", type="primary"):
        all_windows = create_ml_windows(df)

        if len(all_windows) < 50:
            st.warning(
                "Not enough historical data to train and hold out a test period."
            )
        else:
            with st.spinner(
                "Training on historical data and classifying holdout observations..."
            ):
                result = predict_holdout_membership(all_windows)

            p1, p2, p3 = st.columns(3)
            p1.metric("Training Windows", f"{result['train_windows']:,}")
            p2.metric("Holdout Matches", f"{result['matched']:,}")
            p3.metric("Holdout Noise", f"{result['unmatched']:,}")

            if result["train_metrics"]:
                tm = result["train_metrics"]
                m1, m2 = st.columns(2)
                m1.metric(
                    "Training Silhouette Score",
                    f"{tm['silhouette']:.4f}",
                )
                m2.metric(
                    "Training Davies-Bouldin Index",
                    f"{tm['davies_bouldin']:.4f}",
                )

            unmatched = result["holdout_results"][
                ~result["holdout_results"]["MatchesKnownPattern"]
            ]

            if len(unmatched):
                st.markdown("**Holdout observations classified as noise:**")
                st.dataframe(
                    unmatched[
                        [
                            "Window", "Area", "Field",
                            "AlarmCount", "MembershipStrength"
                        ]
                    ]
                    .sort_values("Window", ascending=False)
                    .head(50),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "All holdout observations matched an established cluster."
                )

    st.divider()
    st.caption(
        "This application is a historical decision-support tool. HDBSCAN "
        "identifies statistical patterns in alarm activity. Cluster assignments "
        "and noise classifications do not diagnose equipment failures or "
        "determine operational cause."
    )


if __name__ == "__main__":
    main()
