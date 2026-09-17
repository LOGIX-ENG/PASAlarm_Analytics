import streamlit as st
import pandas as pd


def multiselect_filter(
    label: str,
    values,
    key: str,
):
    options = sorted(
        [
            str(value)
            for value in values
            if pd.notna(value)
        ]
    )

    return st.multiselect(
        label,
        options,
        key=key,
    )


def apply_filters(
    df: pd.DataFrame,
    date_range,
    areas,
    fields,
    assets,
    tags,
    priorities,
    states,
) -> pd.DataFrame:

    result = df.copy()

    if date_range and len(date_range) == 2:

        start_date = date_range[0]
        end_date = date_range[1]

        result = result[
            (result["Date"] >= start_date)
            & (result["Date"] <= end_date)
        ]

    if areas:
        result = result[
            result["Area"].isin(areas)
        ]

    if fields:
        result = result[
            result["Field"].isin(fields)
        ]

    if assets and "Asset" in result.columns:
        result = result[
            result["Asset"].isin(assets)
        ]

    if tags:
        result = result[
            result["Tag"].isin(tags)
        ]

    if priorities:
        result = result[
            result["Priority"].isin(priorities)
        ]

    if states:
        result = result[
            result["Alarm State"].isin(states)
        ]

    return result