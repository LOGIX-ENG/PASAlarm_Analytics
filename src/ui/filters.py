import pandas as pd
import streamlit as st


def multiselect_filter(
    label: str,
    values,
    key: str,
):
    """
    Display a Streamlit multiselect using the supplied values.
    """

    options = sorted(
        [
            value
            for value in values
            if pd.notna(value)
        ],
        key=lambda value: str(value),
    )

    return st.multiselect(
        label,
        options,
        key=key,
    )


def get_filter_options(
    df: pd.DataFrame,
    areas=None,
    fields=None,
    assets=None,
):
    """
    Generate cascading filter options.

    Hierarchy:

        Area
          ↓
        Field
          ↓
        Asset
          ↓
        Tag
    """

    working = df.copy()

    # ------------------------------------------------------------
    # Area
    # ------------------------------------------------------------

    area_options = sorted(
        working["Area"]
        .dropna()
        .unique()
        .tolist(),
        key=lambda value: str(value),
    )

    # ------------------------------------------------------------
    # Field
    # ------------------------------------------------------------

    if areas:
        field_source = working[
            working["Area"].isin(areas)
        ]
    else:
        field_source = working

    field_options = sorted(
        field_source["Field"]
        .dropna()
        .unique()
        .tolist(),
        key=lambda value: str(value),
    )

    # ------------------------------------------------------------
    # Asset
    # ------------------------------------------------------------

    if areas:
        asset_source = working[
            working["Area"].isin(areas)
        ]
    else:
        asset_source = working

    if fields:
        asset_source = asset_source[
            asset_source["Field"].isin(fields)
        ]

    if "Asset" in working.columns:
        asset_options = sorted(
            asset_source["Asset"]
            .dropna()
            .unique()
            .tolist(),
            key=lambda value: str(value),
        )
    else:
        asset_options = []

    # ------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------

    tag_source = working

    if areas:
        tag_source = tag_source[
            tag_source["Area"].isin(areas)
        ]

    if fields:
        tag_source = tag_source[
            tag_source["Field"].isin(fields)
        ]

    if assets and "Asset" in working.columns:
        tag_source = tag_source[
            tag_source["Asset"].isin(assets)
        ]

    tag_options = sorted(
        tag_source["Tag"]
        .dropna()
        .unique()
        .tolist(),
        key=lambda value: str(value),
    )

    return {
        "areas": area_options,
        "fields": field_options,
        "assets": asset_options,
        "tags": tag_options,
    }


def apply_filters(
    df: pd.DataFrame,
    date_range=None,
    areas=None,
    fields=None,
    assets=None,
    tags=None,
    priorities=None,
    states=None,
    alarm_types=None,
    qualities=None,
) -> pd.DataFrame:
    """
    Apply the selected filters to the alarm dataset.

    Filters are applied hierarchically where appropriate:

        Area → Field → Asset → Tag
    """

    result = df.copy()

    # ------------------------------------------------------------
    # Date
    # ------------------------------------------------------------

    if date_range and len(date_range) == 2:
        start_date = pd.to_datetime(
            date_range[0]
        ).date()

        end_date = pd.to_datetime(
            date_range[1]
        ).date()

        if "Date" in result.columns:
            result = result[
                (result["Date"] >= start_date)
                & (result["Date"] <= end_date)
            ]
        else:
            timestamp = pd.to_datetime(
                result["Time / Date"],
                errors="coerce",
            )

            result = result[
                (
                    timestamp.dt.date
                    >= start_date
                )
                & (
                    timestamp.dt.date
                    <= end_date
                )
            ]

    # ------------------------------------------------------------
    # Area
    # ------------------------------------------------------------

    if areas:
        result = result[
            result["Area"].isin(areas)
        ]

    # ------------------------------------------------------------
    # Field
    # ------------------------------------------------------------

    if fields:
        result = result[
            result["Field"].isin(fields)
        ]

    # ------------------------------------------------------------
    # Asset
    # ------------------------------------------------------------

    if assets and "Asset" in result.columns:
        result = result[
            result["Asset"].isin(assets)
        ]

    # ------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------

    if tags:
        result = result[
            result["Tag"].isin(tags)
        ]

    # ------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------

    if priorities:
        result = result[
            result["Priority"].isin(priorities)
        ]

    # ------------------------------------------------------------
    # Alarm State
    # ------------------------------------------------------------

    if states:
        result = result[
            result["Alarm State"].isin(states)
        ]

    # ------------------------------------------------------------
    # Alarm Type
    # ------------------------------------------------------------

    if alarm_types and "Type" in result.columns:
        result = result[
            result["Type"].isin(alarm_types)
        ]

    # ------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------

    if qualities and "Quality" in result.columns:
        result = result[
            result["Quality"].isin(qualities)
        ]

    return result.reset_index(drop=True)