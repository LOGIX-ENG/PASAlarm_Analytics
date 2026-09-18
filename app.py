import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.load_data import (
    load_alarm_data,
    create_data_quality_report,
)
from src.features.temporal_features import (
    add_temporal_features,
)
from src.features.alarm_features import (
    add_alarm_features,
)
from src.features.window_features import (
    create_alarm_windows,
)
from src.analytics.descriptive import (
    alarms_by_area,
    alarms_by_field,
    alarms_by_priority,
    alarms_by_state,
    alarms_by_day,
)
from src.analytics.metrics import (
    calculate_basic_metrics,
)
from src.ml.preprocessing import (
    prepare_ml_data,
)
from src.ml.hdbscan_model import (
    run_hdbscan,
    add_cluster_results,
)
from src.ml.evaluation import (
    evaluate_clusters,
)
from src.ml.visualization import (
    create_pca_projection,
)
from src.ui.filters import (
    get_filter_options,
    apply_filters,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Alarm Analytics",
    page_icon="",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def get_alarm_data():
    df = load_alarm_data()
    df = add_temporal_features(df)
    df = add_alarm_features(df)
    return df


try:
    df = get_alarm_data()

except Exception as exc:
    st.error(
        "The alarm dataset could not be loaded."
    )
    st.exception(exc)
    st.stop()

# ============================================================
# DATA QUALITY
# ============================================================

quality_report = create_data_quality_report(
    df
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Alarm Analytics"
)

st.sidebar.caption(
    "Interactive industrial alarm analysis"
)

# ------------------------------------------------------------
# Date filter
# ------------------------------------------------------------

min_date = df["Date"].min()
max_date = df["Date"].max()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# ------------------------------------------------------------
# Cascading Area → Field → Asset → Tag
# ------------------------------------------------------------

area_options = sorted(
    df["Area"]
    .dropna()
    .unique()
    .tolist()
)

selected_areas = st.sidebar.multiselect(
    "Area",
    area_options,
    key="area_filter",
)

# Field options based on Area

field_source = df

if selected_areas:
    field_source = field_source[
        field_source["Area"].isin(
            selected_areas
        )
    ]

field_options = sorted(
    field_source["Field"]
    .dropna()
    .unique()
    .tolist()
)

selected_fields = st.sidebar.multiselect(
    "Field",
    field_options,
    key="field_filter",
)

# Asset options based on Area + Field

asset_source = field_source

if selected_fields:
    asset_source = asset_source[
        asset_source["Field"].isin(
            selected_fields
        )
    ]

if "Asset" in df.columns:

    asset_options = sorted(
        asset_source["Asset"]
        .dropna()
        .unique()
        .tolist()
    )

else:
    asset_options = []

selected_assets = st.sidebar.multiselect(
    "Asset",
    asset_options,
    key="asset_filter",
)

# Tag options based on Area + Field + Asset

tag_source = asset_source

if selected_assets and "Asset" in df.columns:
    tag_source = tag_source[
        tag_source["Asset"].isin(
            selected_assets
        )
    ]

tag_options = sorted(
    tag_source["Tag"]
    .dropna()
    .unique()
    .tolist()
)

selected_tags = st.sidebar.multiselect(
    "Tag",
    tag_options,
    key="tag_filter",
)

# ------------------------------------------------------------
# Other filters
# ------------------------------------------------------------

priority_options = sorted(
    df["Priority"]
    .dropna()
    .unique()
    .tolist()
)

selected_priorities = st.sidebar.multiselect(
    "Priority",
    priority_options,
    key="priority_filter",
)

state_options = sorted(
    df["Alarm State"]
    .dropna()
    .unique()
    .tolist()
)

selected_states = st.sidebar.multiselect(
    "Alarm State",
    state_options,
    key="state_filter",
)

type_options = sorted(
    df["Type"]
    .dropna()
    .unique()
    .tolist()
)

selected_types = st.sidebar.multiselect(
    "Type",
    type_options,
    key="type_filter",
)

quality_options = sorted(
    df["Quality"]
    .dropna()
    .unique()
    .tolist()
)

selected_qualities = st.sidebar.multiselect(
    "Quality",
    quality_options,
    key="quality_filter",
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = apply_filters(
    df=df,
    date_range=date_range,
    areas=selected_areas,
    fields=selected_fields,
    assets=selected_assets,
    tags=selected_tags,
    priorities=selected_priorities,
    states=selected_states,
    alarm_types=selected_types,
    qualities=selected_qualities,
)

# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Alarm Explorer",
        "Machine Learning",
        "Data Quality",
        "Methodology",
    ],
)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title(
        "Industrial Alarm Analytics"
    )

    st.write(
        "Interactive analysis of historical industrial "
        "alarm activity using descriptive analytics and "
        "HDBSCAN-based pattern discovery."
    )

    metrics = calculate_basic_metrics(
        filtered_df
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Alarm Events",
        f"{metrics['total_events']:,}",
    )

    col2.metric(
        "Areas",
        f"{metrics['unique_areas']:,}",
    )

    col3.metric(
        "Fields",
        f"{metrics['unique_fields']:,}",
    )

    col4.metric(
        "Assets",
        f"{metrics['unique_assets']:,}",
    )

    col5.metric(
        "Tags",
        f"{metrics['unique_tags']:,}",
    )

    st.divider()

    # --------------------------------------------------------
    # Alarm trend
    # --------------------------------------------------------

    daily = alarms_by_day(
        filtered_df
    )

    if not daily.empty:
        fig = px.line(
            daily,
            x="Date",
            y="Alarm Count",
            title="Alarm Activity Over Time",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Alarm Count",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Area and Field
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        area_data = alarms_by_area(
            filtered_df
        )

        if not area_data.empty:
            fig = px.bar(
                area_data,
                x="Area",
                y="Alarm Count",
                title="Alarm Activity by Area",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with col2:

        field_data = alarms_by_field(
            filtered_df
        )

        if not field_data.empty:
            fig = px.bar(
                field_data,
                x="Field",
                y="Alarm Count",
                color="Area",
                title="Alarm Activity by Field",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # Priority and State
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        priority_data = alarms_by_priority(
            filtered_df
        )

        if not priority_data.empty:
            fig = px.bar(
                priority_data,
                x="Priority",
                y="Alarm Count",
                title="Alarm Activity by Priority",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with col2:

        state_data = alarms_by_state(
            filtered_df
        )

        if not state_data.empty:
            fig = px.bar(
                state_data,
                x="Alarm State",
                y="Alarm Count",
                title="Alarm Activity by Alarm State",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# ALARM EXPLORER
# ============================================================

elif page == "Alarm Explorer":

    st.title(
        "Alarm Explorer"
    )

    st.write(
        "Use the filters to investigate individual alarm "
        "events across the Area → Field → Asset → Tag hierarchy."
    )

    st.metric(
        "Filtered Events",
        f"{len(filtered_df):,}",
    )

    # --------------------------------------------------------
    # Download filtered dataset
    # --------------------------------------------------------

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Results",
        data=csv_data,
        file_name="filtered_alarm_results.csv",
        mime="text/csv",
    )

    st.divider()

    # --------------------------------------------------------
    # Display data
    # --------------------------------------------------------

    display_columns = [
        "Time / Date",
        "Tag",
        "Priority",
        "Type",
        "Quality",
        "Alarm State",
        "Area",
        "Field",
    ]

    if "Asset" in filtered_df.columns:
        display_columns.insert(
            7,
            "Asset",
        )

    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.title(
        "HDBSCAN Pattern Discovery"
    )

    st.write(
        "HDBSCAN analyzes fixed time-window observations "
        "to identify recurring alarm-activity patterns and "
        "observations that do not fit the learned density structure."
    )

    st.info(
        "HDBSCAN noise observations are statistically unusual "
        "relative to the modeled data. They are candidates for "
        "engineering investigation and do not automatically "
        "represent equipment faults or process failures."
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        window_label = st.selectbox(
            "Time Window",
            [
                "5 minutes",
                "10 minutes",
                "15 minutes",
                "30 minutes",
            ],
        )

    window_map = {
        "5 minutes": "5min",
        "10 minutes": "10min",
        "15 minutes": "15min",
        "30 minutes": "30min",
    }

    selected_window = window_map[
        window_label
    ]

    with col2:

        min_cluster_size = st.slider(
            "Minimum Cluster Size",
            min_value=2,
            max_value=100,
            value=10,
            step=1,
        )

    # --------------------------------------------------------
    # Run button
    # --------------------------------------------------------

    run_model = st.button(
        "Run HDBSCAN",
        type="primary",
    )

    if run_model:

        if filtered_df.empty:
            st.warning(
                "No alarm events match the current filters."
            )
            st.stop()

        with st.spinner(
                "Creating alarm windows and running HDBSCAN..."
        ):

            window_df = create_alarm_windows(
                filtered_df,
                window=selected_window,
            )

            if len(window_df) < min_cluster_size:
                st.warning(
                    "There are not enough Area/Field windows "
                    "to run HDBSCAN with the selected minimum "
                    "cluster size."
                )
                st.stop()

            (
                scaled_features,
                scaler,
                feature_names,
                feature_data,
            ) = prepare_ml_data(window_df)

            clusterer = run_hdbscan(
                scaled_features,
                min_cluster_size=min_cluster_size,
            )

            cluster_df = add_cluster_results(
                window_df,
                clusterer,
            )

            evaluation = evaluate_clusters(
                scaled_features,
                cluster_df["Cluster"].values,
            )

            projection_df = create_pca_projection(
                scaled_features,
                cluster_df,
            )

            st.session_state[
                "cluster_df"
            ] = cluster_df

            st.session_state[
                "projection_df"
            ] = projection_df

            st.session_state[
                "evaluation"
            ] = evaluation

            st.session_state[
                "feature_names"
            ] = feature_names

            st.session_state[
                "selected_window"
            ] = selected_window

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if "cluster_df" in st.session_state:

        cluster_df = st.session_state[
            "cluster_df"
        ]

        projection_df = st.session_state[
            "projection_df"
        ]

        evaluation = st.session_state[
            "evaluation"
        ]

        feature_names = st.session_state[
            "feature_names"
        ]

        st.subheader(
            "Model Results"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Clusters",
            evaluation["cluster_count"],
        )

        col2.metric(
            "Noise Observations",
            evaluation["noise_count"],
        )

        col3.metric(
            "Noise %",
            f"{evaluation['noise_percentage']:.1f}%",
        )

        if evaluation[
            "silhouette_score"
        ] is not None:

            col4.metric(
                "Silhouette",
                f"{evaluation['silhouette_score']:.3f}",
            )

        else:

            col4.metric(
                "Silhouette",
                "N/A",
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        st.subheader(
            "Evaluation"
        )

        evaluation_col1, evaluation_col2 = (
            st.columns(2)
        )

        with evaluation_col1:

            if evaluation[
                "silhouette_score"
            ] is not None:

                st.write(
                    "**Silhouette Score:** "
                    f"{evaluation['silhouette_score']:.3f}"
                )

            else:

                st.write(
                    "**Silhouette Score:** N/A"
                )

        with evaluation_col2:

            if evaluation[
                "davies_bouldin_score"
            ] is not None:

                st.write(
                    "**Davies-Bouldin Score:** "
                    f"{evaluation['davies_bouldin_score']:.3f}"
                )

            else:

                st.write(
                    "**Davies-Bouldin Score:** N/A"
                )

        st.caption(
            "Silhouette and Davies-Bouldin scores are calculated "
            "using non-noise observations."
        )

        # ----------------------------------------------------
        # PCA visualization
        # ----------------------------------------------------

        st.subheader(
            "Cluster Visualization"
        )

        fig = px.scatter(
            projection_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            hover_data=[
                "Window",
                "Area",
                "Field",
                "AlarmCount",
                "UniqueTags",
                "ActiveCount",
                "AckedCount",
                "NormalCount",
            ],
            title=(
                "HDBSCAN Alarm Activity Clusters "
                "(PCA Projection)"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "PCA is used only to visualize the feature space. "
            "HDBSCAN is run on the standardized feature set."
        )

        # ----------------------------------------------------
        # Feature information
        # ----------------------------------------------------

        with st.expander(
                "ML Features"
        ):

            st.write(
                "Features used by HDBSCAN:"
            )

            for feature in feature_names:
                st.write(
                    f"- {feature}"
                )

        # ----------------------------------------------------
        # Cluster summary
        # ----------------------------------------------------

        st.subheader(
            "Cluster Summary"
        )

        cluster_summary = (
            cluster_df
            .groupby("Cluster")
            .agg(
                Observations=("Cluster", "size"),
                AverageAlarmCount=(
                    "AlarmCount",
                    "mean",
                ),
                AverageUniqueTags=(
                    "UniqueTags",
                    "mean",
                ),
                AveragePriority=(
                    "AveragePriority",
                    "mean",
                ),
            )
            .reset_index()
        )

        cluster_summary = (
            cluster_summary
            .sort_values("Cluster")
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Inspect cluster
        # ----------------------------------------------------

        st.subheader(
            "Inspect Cluster"
        )

        available_clusters = sorted(
            cluster_df["Cluster"]
            .unique()
            .tolist()
        )

        selected_cluster = st.selectbox(
            "Select Cluster",
            available_clusters,
        )

        selected_cluster_df = (
            cluster_df[
                cluster_df["Cluster"]
                == selected_cluster
                ]
            .sort_values("Window")
        )

        st.write(
            f"Observations in selected cluster: "
            f"{len(selected_cluster_df):,}"
        )

        st.dataframe(
            selected_cluster_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.title(
        "Data Quality"
    )

    st.write(
        "Validation and quality indicators for the dataset "
        "used by the application."
    )

    # --------------------------------------------------------
    # Primary metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Records",
        f"{quality_report['total_records']:,}",
    )

    col2.metric(
        "Valid Timestamps",
        f"{quality_report['valid_timestamps']:,}",
    )

    col3.metric(
        "Invalid Timestamps",
        f"{quality_report['invalid_timestamps']:,}",
    )

    col4.metric(
        "Invalid Priorities",
        f"{quality_report['invalid_priorities']:,}",
    )

    st.divider()

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    st.subheader(
        "Dataset Coverage"
    )

    start_date = quality_report[
        "date_range_start"
    ]

    end_date = quality_report[
        "date_range_end"
    ]

    if start_date is not None:
        st.write(
            f"**Date Range:** "
            f"{start_date:%Y-%m-%d %H:%M:%S} "
            f"through "
            f"{end_date:%Y-%m-%d %H:%M:%S}"
        )

    st.write(
        f"**Areas:** "
        f"{quality_report['unique_areas']:,}"
    )

    st.write(
        f"**Fields:** "
        f"{quality_report['unique_fields']:,}"
    )

    st.write(
        f"**Assets:** "
        f"{quality_report['unique_assets']:,}"
    )

    st.write(
        f"**Tags:** "
        f"{quality_report['unique_tags']:,}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    missing_df = pd.DataFrame(
        [
            {
                "Column": column,
                "Missing Values": count,
            }
            for column, count
            in quality_report[
            "missing_values"
        ].items()
        ]
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Alarm states
    # --------------------------------------------------------

    st.subheader(
        "Alarm State Validation"
    )

    observed_states = quality_report[
        "observed_alarm_states"
    ]

    unexpected_states = quality_report[
        "unexpected_alarm_states"
    ]

    st.write(
        "**Observed States:** "
        + ", ".join(observed_states)
        if observed_states
        else "**Observed States:** None"
    )

    if unexpected_states:

        st.warning(
            "Unexpected alarm states detected: "
            + ", ".join(unexpected_states)
        )

    else:

        st.success(
            "No unexpected alarm states were detected."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    st.subheader(
        "Required Columns"
    )

    required_columns_status = pd.DataFrame(
        {
            "Column": [
                "Time / Date",
                "Tag",
                "Priority",
                "Type",
                "Quality",
                "Alarm State",
                "Area",
                "Field",
            ],
            "Present": [
                column in df.columns
                for column in [
                    "Time / Date",
                    "Tag",
                    "Priority",
                    "Type",
                    "Quality",
                    "Alarm State",
                    "Area",
                    "Field",
                ]
            ],
        }
    )

    st.dataframe(
        required_columns_status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.title(
        "Methodology"
    )

    st.subheader(
        "Business Problem"
    )

    st.write(
        "Industrial SCADA systems can generate large volumes "
        "of alarm events across Areas, Fields, Assets, and "
        "individual alarm tags. Manual examination of historical "
        "events makes it difficult to identify recurring patterns "
        "and unusual concentrations of alarm activity."
    )

    st.subheader(
        "Analytical Approach"
    )

    st.write(
        "The application combines descriptive analytics with "
        "unsupervised machine learning."
    )

    st.markdown(
        """
        **Descriptive analytics**

        Historical alarm events are summarized by:
        - Area
        - Field
        - Asset
        - Tag
        - Priority
        - Alarm State
        - Time

        **HDBSCAN**

        Alarm events are aggregated into fixed time windows.
        Each machine-learning observation represents an Area/Field
        combination within the selected time window.

        HDBSCAN identifies density-based clusters and observations
        that do not fit the learned density structure.
        """
    )

    st.subheader(
        "ML Observation"
    )

    st.write(
        "Each HDBSCAN observation represents one Area + Field "
        "combination within a selected time window."
    )

    st.code(
        """
Alarm Events
     ↓
Time-window aggregation
     ↓
Area + Field observations
     ↓
Feature engineering
     ↓
StandardScaler
     ↓
HDBSCAN
     ↓
Clusters + noise observations
     ↓
Interactive investigation
        """
    )

    st.subheader(
        "Features"
    )

    st.write(
        "The current model can use the following features:"
    )

    features = [
        "AlarmCount",
        "UniqueTags",
        "AveragePriority",
        "MaximumPriority",
        "ActiveCount",
        "AckedCount",
        "NormalCount",
        "HighPriorityCount",
        "AlarmRatePerMinute",
    ]

    for feature in features:
        st.write(
            f"- {feature}"
        )

    st.subheader(
        "Alarm State Interpretation"
    )

    st.write(
        "`Active` represents a recorded active alarm state. "
        "`Acked` represents a SCADA-recorded acknowledged state. "
        "`Normal` represents a recorded normal state."
    )

    st.write(
        "A Normal record does not independently establish whether "
        "the alarm had previously been acknowledged or whether an "
        "operator physically intervened."
    )

    st.subheader(
        "HDBSCAN Noise"
    )

    st.write(
        "HDBSCAN assigns a cluster label of -1 to observations "
        "classified as noise. These observations do not fit the "
        "learned density structure under the selected parameters."
    )

    st.write(
        "Noise observations are candidates for engineering "
        "investigation and should not automatically be interpreted "
        "as equipment failures, process abnormalities, or operator error."
    )

    st.subheader(
        "PCA Visualization"
    )

    st.write(
        "Principal Component Analysis (PCA) is used to project "
        "the standardized feature space into two dimensions for "
        "visualization. HDBSCAN itself is performed on the "
        "standardized feature set rather than the PCA projection."
    )

    st.subheader(
        "ISA-18.2 Positioning"
    )

    st.write(
        "The application is intended to support monitoring and "
        "assessment of industrial alarm activity in an "
        "ISA-18.2-oriented context. It should not be represented "
        "as an ISA-18.2 compliance certification tool."
    )

    st.subheader(
        "Data Security"
    )

    st.write(
        "The dataset used by this application has been sanitized "
        "and anonymized for educational use. Original facility, "
        "asset, and tag identifiers are not exposed in the public "
        "application. Proprietary source data is excluded from "
        "the public repository. Credentials and application "
        "secrets are not stored in source code."
    )

    st.subheader(
        "Limitations"
    )

    st.markdown(
        """
        - The dataset does not provide complete operator-response history.
        - HDBSCAN noise does not inherently represent equipment failure.
        - The clustering approach is unsupervised and does not use labeled fault data.
        - Clustering results depend on feature selection and model parameters.
        - Results should be interpreted as decision-support information rather than automated diagnosis.
        """
    )