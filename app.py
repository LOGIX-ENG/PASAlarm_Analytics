import streamlit as st
import pandas as pd
import numpy as np
import hdbscan

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# APPLICATION SETTINGS
# ============================================================

st.set_page_config(
    page_title="Industrial Alarm Analytics",
    page_icon="",
    layout="wide"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    """Load and prepare the alarm dataset."""

    df = pd.read_csv("data/alarms.csv")

    # Convert timestamp
    df["Time / Date"] = pd.to_datetime(
        df["Time / Date"],
        errors="coerce"
    )

    # Convert priority
    df["Priority"] = pd.to_numeric(
        df["Priority"],
        errors="coerce"
    )

    # Remove records without a valid timestamp
    df = df.dropna(
        subset=["Time / Date"]
    ).copy()

    # Sort chronologically
    df = df.sort_values(
        "Time / Date"
    )

    return df


# ============================================================
# CREATE MACHINE-LEARNING WINDOWS
# ============================================================

def create_ml_windows(df):
    """
    Convert alarm events into five-minute Area/Field
    observations for HDBSCAN.
    """

    data = df.copy()

    # Five-minute time window
    data["Window"] = (
        data["Time / Date"]
        .dt.floor("5min")
    )

    # Basic alarm-state indicators
    data["ActiveFlag"] = (
        data["Alarm State"]
        .astype(str)
        .str.lower()
        .eq("active")
        .astype(int)
    )

    data["AckedFlag"] = (
        data["Alarm State"]
        .astype(str)
        .str.lower()
        .eq("acked")
        .astype(int)
    )

    data["NormalFlag"] = (
        data["Alarm State"]
        .astype(str)
        .str.lower()
        .eq("normal")
        .astype(int)
    )

    # High-priority indicator
    data["HighPriorityFlag"] = (
        data["Priority"]
        .fillna(0)
        .ge(500)
        .astype(int)
    )

    # Aggregate alarm events
    grouped = (
        data
        .groupby(
            ["Window", "Area", "Field"],
            dropna=False
        )
        .agg(
            AlarmCount=("Tag", "count"),
            UniqueTags=("Tag", "nunique"),
            AveragePriority=("Priority", "mean"),
            MaximumPriority=("Priority", "max"),
            ActiveCount=("ActiveFlag", "sum"),
            AckedCount=("AckedFlag", "sum"),
            NormalCount=("NormalFlag", "sum"),
            HighPriorityCount=(
                "HighPriorityFlag",
                "sum"
            )
        )
        .reset_index()
    )

    # Five-minute window = 5 minutes
    grouped["AlarmRatePerMinute"] = (
        grouped["AlarmCount"] / 5
    )

    return grouped


# ============================================================
# RUN HDBSCAN
# ============================================================

def run_hdbscan(window_data):
    """Run the fixed baseline HDBSCAN configuration."""

    feature_columns = [
        "AlarmCount",
        "UniqueTags",
        "AveragePriority",
        "MaximumPriority",
        "ActiveCount",
        "AckedCount",
        "NormalCount",
        "HighPriorityCount",
        "AlarmRatePerMinute"
    ]

    # Replace missing numeric values
    features = (
        window_data[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Standardize features
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    # Fixed project configuration
    model = hdbscan.HDBSCAN(
        min_cluster_size=10
    )

    labels = model.fit_predict(
        scaled_features
    )

    results = window_data.copy()

    results["Cluster"] = labels

    return results, scaled_features


# ============================================================
# APPLICATION
# ============================================================

st.title(
    "Industrial Alarm Analytics"
)

st.write(
    "Interactive analysis of historical industrial "
    "alarm activity using descriptive analytics and "
    "unsupervised machine learning."
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# BASIC DATA SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Alarm Records",
    f"{len(df):,}"
)

col2.metric(
    "Areas",
    df["Area"].nunique()
)

col3.metric(
    "Fields",
    df["Field"].nunique()
)


# ============================================================
# FILTERS
# ============================================================

st.subheader("Explore the Data")

filter_col1, filter_col2 = st.columns(2)


with filter_col1:

    area_options = (
        ["All"]
        + sorted(
            df["Area"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    selected_area = st.selectbox(
        "Area",
        area_options
    )


with filter_col2:

    priority_options = (
        ["All"]
        + sorted(
            df["Priority"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    selected_priority = st.selectbox(
        "Priority",
        priority_options
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_area != "All":

    filtered_df = filtered_df[
        filtered_df["Area"].astype(str)
        == selected_area
    ]


if selected_priority != "All":

    filtered_df = filtered_df[
        filtered_df["Priority"]
        == selected_priority
    ]


st.write(
    f"Showing **{len(filtered_df):,}** "
    f"of **{len(df):,}** alarm records."
)


# ============================================================
# VISUAL 1
# ============================================================

st.subheader("1. Alarm Events by Area")

area_counts = (
    filtered_df["Area"]
    .value_counts()
    .sort_values(ascending=False)
)

st.bar_chart(
    area_counts
)


# ============================================================
# VISUAL 2
# ============================================================

st.subheader("2. Alarm Events Over Time")

daily_counts = (
    filtered_df
    .set_index("Time / Date")
    .resample("D")
    .size()
)

st.line_chart(
    daily_counts
)


# ============================================================
# FILTERED DATA
# ============================================================

st.subheader("Filtered Alarm Records")

display_columns = [
    column
    for column in [
        "Time / Date",
        "Tag",
        "Priority",
        "Type",
        "Quality",
        "Alarm State",
        "Area",
        "Field"
    ]
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[
        display_columns
    ].head(500),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MACHINE LEARNING
# ============================================================

st.subheader(
    "3. HDBSCAN Pattern Discovery"
)

st.write(
    "HDBSCAN groups similar five-minute Area/Field "
    "alarm-activity observations. Observations classified "
    "as noise do not fit the learned density structure."
)


if st.button(
    "Run HDBSCAN",
    type="primary"
):

    if len(filtered_df) < 20:

        st.warning(
            "There is not enough filtered data "
            "to perform the analysis."
        )

    else:

        with st.spinner(
            "Running HDBSCAN..."
        ):

            window_data = create_ml_windows(
                filtered_df
            )

            if len(window_data) < 20:

                st.warning(
                    "The selected data does not "
                    "produce enough five-minute "
                    "observations for analysis."
                )

            else:

                results, scaled_features = (
                    run_hdbscan(
                        window_data
                    )
                )

                # Number of clusters
                cluster_labels = set(
                    results["Cluster"]
                )

                cluster_count = len(
                    cluster_labels - {-1}
                )

                # Noise
                noise_count = int(
                    (
                        results["Cluster"] == -1
                    ).sum()
                )

                noise_percent = (
                    noise_count
                    / len(results)
                    * 100
                )

                # Results summary
                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )

                result_col1.metric(
                    "Clusters",
                    cluster_count
                )

                result_col2.metric(
                    "Noise Observations",
                    noise_count
                )

                result_col3.metric(
                    "Noise %",
                    f"{noise_percent:.2f}%"
                )

                # ------------------------------------------------
                # PCA FOR VISUALIZATION ONLY
                # ------------------------------------------------

                pca = PCA(
                    n_components=2
                )

                coordinates = (
                    pca.fit_transform(
                        scaled_features
                    )
                )

                plot_data = pd.DataFrame(
                    {
                        "PC1": coordinates[:, 0],
                        "PC2": coordinates[:, 1],
                        "Cluster": (
                            results["Cluster"]
                            .astype(str)
                        )
                    }
                )

                st.scatter_chart(
                    plot_data,
                    x="PC1",
                    y="PC2",
                    color="Cluster"
                )

                st.caption(
                    "Each point represents one five-minute "
                    "Area/Field observation. PCA is used "
                    "only to display the HDBSCAN results "
                    "in two dimensions."
                )


# ============================================================
# PROJECT NOTE
# ============================================================

st.divider()

st.caption(
    "This application is a historical decision-support "
    "tool. HDBSCAN identifies statistical patterns and "
    "does not diagnose equipment failures or automatically "
    "determine the cause of an alarm."
)