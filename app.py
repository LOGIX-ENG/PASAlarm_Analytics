import streamlit as st

from src.analytics.descriptive import (
    alarms_by_area,
    alarms_by_field,
    alarms_by_day,
    alarms_by_priority,
    alarms_by_state,
)
from src.analytics.metrics import (
    calculate_basic_metrics,
)
from src.data.load_data import load_alarm_data
from src.features.alarm_features import (
    add_alarm_features,
)
from src.features.temporal_features import (
    add_temporal_features,
)
from src.features.window_features import (
    create_alarm_windows,
)
from src.ml.evaluation import (
    evaluate_clusters,
)
from src.ml.hdbscan_model import (
    run_hdbscan,
    add_cluster_results,
)
from src.ml.preprocessing import (
    prepare_ml_data,
)
from src.ml.visualization import (
    create_pca_projection,
)
from src.ui.charts import (
    area_chart,
    field_chart,
    alarm_trend_chart,
    priority_chart,
    state_chart,
    cluster_chart,
)
from src.ui.filters import (
    multiselect_filter,
    apply_filters,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Industrial Alarm Analytics",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def get_data():

    df = load_alarm_data()

    df = add_temporal_features(df)

    df = add_alarm_features(df)

    return df


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

def main():

    st.title(
        "Industrial Alarm Analytics"
    )

    st.caption(
        "Descriptive analytics and density-based "
        "machine learning for historical industrial "
        "alarm activity."
    )

    try:

        df = get_data()

    except Exception as error:

        st.error(
            f"Unable to load alarm data: {error}"
        )

        st.stop()


    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    st.sidebar.header(
        "Analysis Filters"
    )

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(
            min_date,
            max_date,
        ),
        min_value=min_date,
        max_value=max_date,
    )

    areas = multiselect_filter(
        "Area",
        df["Area"].unique(),
        "area_filter",
    )

    filtered_for_fields = df.copy()

    if areas:
        filtered_for_fields = (
            filtered_for_fields[
                filtered_for_fields["Area"].isin(
                    areas
                )
            ]
        )

    fields = multiselect_filter(
        "Field",
        filtered_for_fields["Field"].unique(),
        "field_filter",
    )

    filtered_for_assets = (
        filtered_for_fields.copy()
    )

    if fields:
        filtered_for_assets = (
            filtered_for_assets[
                filtered_for_assets["Field"].isin(
                    fields
                )
            ]
        )

    assets = []

    if "Asset" in df.columns:

        assets = multiselect_filter(
            "Asset",
            filtered_for_assets["Asset"].unique(),
            "asset_filter",
        )

    tags = multiselect_filter(
        "Alarm Tag",
        filtered_for_assets["Tag"].unique(),
        "tag_filter",
    )

    priorities = st.sidebar.multiselect(
        "Priority",
        sorted(
            df["Priority"]
            .dropna()
            .unique()
        ),
    )

    states = st.sidebar.multiselect(
        "Alarm State",
        sorted(
            df["Alarm State"]
            .dropna()
            .unique()
        ),
    )


    # -----------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------

    filtered_df = apply_filters(
        df=df,
        date_range=date_range,
        areas=areas,
        fields=fields,
        assets=assets,
        tags=tags,
        priorities=priorities,
        states=states,
    )


    # -----------------------------------------------------
    # NAVIGATION
    # -----------------------------------------------------

    page = st.sidebar.radio(
        "Application",
        [
            "Overview",
            "Alarm Explorer",
            "Machine Learning",
            "Methodology",
        ],
    )


    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    if page == "Overview":

        st.header(
            "Alarm System Overview"
        )

        metrics = calculate_basic_metrics(
            filtered_df
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Alarm Events",
            f"{metrics['total_events']:,}",
        )

        col2.metric(
            "Unique Tags",
            f"{metrics['unique_tags']:,}",
        )

        col3.metric(
            "Areas",
            f"{metrics['unique_areas']:,}",
        )

        col4.metric(
            "Fields",
            f"{metrics['unique_fields']:,}",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            area_data = alarms_by_area(
                filtered_df
            )

            st.plotly_chart(
                area_chart(area_data),
                use_container_width=True,
            )

        with col2:

            state_data = alarms_by_state(
                filtered_df
            )

            st.plotly_chart(
                state_chart(state_data),
                use_container_width=True,
            )

        trend_data = alarms_by_day(
            filtered_df
        )

        st.plotly_chart(
            alarm_trend_chart(trend_data),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            field_data = alarms_by_field(
                filtered_df
            )

            st.plotly_chart(
                field_chart(field_data),
                use_container_width=True,
            )

        with col2:

            priority_data = alarms_by_priority(
                filtered_df
            )

            st.plotly_chart(
                priority_chart(priority_data),
                use_container_width=True,
            )


    # -----------------------------------------------------
    # ALARM EXPLORER
    # -----------------------------------------------------

    elif page == "Alarm Explorer":

        st.header(
            "Alarm Explorer"
        )

        st.write(
            "Explore individual alarm events using "
            "the hierarchy Area → Field → Asset → Tag."
        )

        st.metric(
            "Filtered Alarm Events",
            f"{len(filtered_df):,}",
        )

        display_columns = [
            column
            for column in [
                "Time / Date",
                "Area",
                "Field",
                "Asset",
                "Tag",
                "Priority",
                "Type",
                "Quality",
                "Alarm State",
            ]
            if column in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[
                display_columns
            ].sort_values(
                "Time / Date",
                ascending=False,
            ),
            use_container_width=True,
            height=500,
        )


    # -----------------------------------------------------
    # MACHINE LEARNING
    # -----------------------------------------------------

    elif page == "Machine Learning":

        st.header(
            "Machine Learning"
        )

        st.write(
            """
            HDBSCAN is applied to 5-minute Area/Field
            alarm activity windows. The algorithm identifies
            dense patterns in the feature space without
            requiring a predefined number of clusters.
            """
        )

        st.info(
            "A cluster represents a recurring pattern of "
            "alarm activity. A noise point (-1) represents "
            "an observation that does not fit a learned "
            "density pattern. Noise is not automatically "
            "evidence of an equipment fault."
        )

        # ---------------------------------------------
        # MODEL PARAMETERS
        # ---------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            window_size = st.selectbox(
                "Time Window",
                [
                    "5min",
                    "10min",
                    "15min",
                    "30min",
                ],
                index=0,
            )

        with col2:

            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=3,
                max_value=50,
                value=10,
            )

        if st.button(
            "Run HDBSCAN Analysis",
            type="primary",
        ):

            with st.spinner(
                "Running machine learning analysis..."
            ):

                window_df = create_alarm_windows(
                    filtered_df,
                    window=window_size,
                )

                if len(window_df) < min_cluster_size:

                    st.warning(
                        "There are not enough Area/Field "
                        "windows for the selected minimum "
                        "cluster size."
                    )

                    st.stop()

                (
                    scaled_features,
                    scaler,
                    feature_names,
                ) = prepare_ml_data(
                    window_df
                )

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

                projection_df = (
                    create_pca_projection(
                        scaled_features,
                        cluster_df,
                    )
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


        # ---------------------------------------------
        # DISPLAY MODEL RESULTS
        # ---------------------------------------------

        if "evaluation" in st.session_state:

            evaluation = (
                st.session_state[
                    "evaluation"
                ]
            )

            cluster_df = (
                st.session_state[
                    "cluster_df"
                ]
            )

            projection_df = (
                st.session_state[
                    "projection_df"
                ]
            )

            st.divider()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Clusters",
                evaluation[
                    "cluster_count"
                ],
            )

            col2.metric(
                "Noise Windows",
                evaluation[
                    "noise_count"
                ],
            )

            col3.metric(
                "Noise %",
                f"{evaluation['noise_percentage']:.1f}%",
            )

            if (
                evaluation[
                    "silhouette_score"
                ]
                is not None
            ):

                col4.metric(
                    "Silhouette Score",
                    f"{evaluation['silhouette_score']:.3f}",
                )

            else:

                col4.metric(
                    "Silhouette Score",
                    "N/A",
                )

            st.plotly_chart(
                cluster_chart(
                    projection_df
                ),
                use_container_width=True,
            )

            st.subheader(
                "Cluster Summary"
            )

            cluster_summary = (
                cluster_df
                .groupby("Cluster")
                .agg(
                    Windows=("Cluster", "size"),
                    AverageAlarms=(
                        "AlarmCount",
                        "mean",
                    ),
                    AverageTags=(
                        "UniqueTags",
                        "mean",
                    ),
                    AverageActive=(
                        "ActiveCount",
                        "mean",
                    ),
                )
                .reset_index()
            )

            st.dataframe(
                cluster_summary,
                use_container_width=True,
            )

            st.subheader(
                "Explore Cluster"
            )

            available_clusters = sorted(
                cluster_df["Cluster"]
                .unique()
            )

            selected_cluster = st.selectbox(
                "Select a cluster",
                available_clusters,
            )

            selected_data = cluster_df[
                cluster_df["Cluster"]
                == selected_cluster
            ]

            st.dataframe(
                selected_data.sort_values(
                    "Window",
                    ascending=False,
                ),
                use_container_width=True,
                height=400,
            )


    # -----------------------------------------------------
    # METHODOLOGY
    # -----------------------------------------------------

    elif page == "Methodology":

        st.header(
            "Methodology"
        )

        st.subheader(
            "Business Problem"
        )

        st.write(
            """
            Industrial SCADA systems can generate large
            volumes of alarm events across Areas, Fields,
            Assets, and individual alarm points. Manual
            analysis of historical alarm activity can make
            it difficult to identify recurring patterns,
            concentrations, and unusual activity.
            """
        )

        st.subheader(
            "Descriptive Analytics"
        )

        st.write(
            """
            Descriptive analytics summarizes what occurred
            in the historical alarm data. The application
            examines alarm frequency, priority, alarm state,
            time-based activity, Areas, Fields, Assets, and
            individual alarm Tags.
            """
        )

        st.subheader(
            "Machine Learning"
        )

        st.write(
            """
            HDBSCAN is used as the non-descriptive machine
            learning method. Alarm events are aggregated
            into fixed time windows for each Area and Field.
            Numerical activity features are standardized
            before clustering.
            """
        )

        st.subheader(
            "Hierarchy"
        )

        st.code(
            """
Area
  ↓
Field
  ↓
Asset
  ↓
Alarm Tag
            """
        )

        st.subheader(
            "Alarm State Interpretation"
        )

        st.write(
            """
            Active indicates that the alarm bit is active.
            Acked represents an acknowledgment recorded by
            SCADA. Normal indicates that the alarm condition
            returned to normal. A Normal record does not by
            itself establish whether an operator previously
            acknowledged or acted on the alarm.
            """
        )

        st.subheader(
            "HDBSCAN Noise"
        )

        st.write(
            """
            HDBSCAN labels observations that do not belong
            to a sufficiently dense cluster as noise (-1).
            In this application, noise represents unusual
            alarm-activity patterns relative to the learned
            dataset. It should not be interpreted directly
            as an equipment failure or process abnormality.
            """)

        st.subheader(
            "ISA-18.2 Positioning"
        )

        st.write(
            """
            The application is designed to support
            ISA-18.2-oriented monitoring and assessment of
            historical alarm activity. It does not claim
            that the underlying alarm system is ISA-18.2
            compliant.
            """)


if __name__ == "__main__":
    main()