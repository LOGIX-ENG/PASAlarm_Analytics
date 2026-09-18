import plotly.express as px


def area_chart(df):

    fig = px.bar(
        df,
        x="Area",
        y="Alarm Count",
        title="Alarm Activity by Area",
    )

    fig.update_layout(
        xaxis_title="Area",
        yaxis_title="Alarm Count",
    )

    return fig


def field_chart(df):

    fig = px.bar(
        df,
        x="Field",
        y="Alarm Count",
        color="Area",
        title="Alarm Activity by Field",
    )

    fig.update_layout(
        xaxis_title="Field",
        yaxis_title="Alarm Count",
    )

    return fig


def alarm_trend_chart(df):

    fig = px.line(
        df,
        x="Date",
        y="Alarm Count",
        title="Alarm Activity Over Time",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Alarm Count",
    )

    return fig


def priority_chart(df):

    fig = px.bar(
        df,
        x="Priority",
        y="Alarm Count",
        title="Alarm Activity by Priority",
    )

    fig.update_layout(
        xaxis_title="Priority",
        yaxis_title="Alarm Count",
    )

    return fig


def state_chart(df):

    fig = px.bar(
        df,
        x="Alarm State",
        y="Alarm Count",
        title="Alarm Activity by Alarm State",
    )

    fig.update_layout(
        xaxis_title="Alarm State",
        yaxis_title="Alarm Count",
    )

    return fig


def type_chart(df):

    fig = px.bar(
        df,
        x="Type",
        y="Alarm Count",
        title="Alarm Activity by Type",
    )

    return fig


def cluster_chart(df):

    fig = px.scatter(
        df,
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

    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
    )

    return fig