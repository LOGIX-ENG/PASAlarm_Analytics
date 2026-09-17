import pandas as pd


def calculate_basic_metrics(df: pd.DataFrame) -> dict:
    """Calculate high-level descriptive metrics."""

    return {
        "total_events": len(df),

        "unique_tags": (
            df["Tag"].nunique()
            if "Tag" in df.columns
            else 0
        ),

        "unique_assets": (
            df["Asset"].nunique()
            if "Asset" in df.columns
            else 0
        ),

        "unique_areas": (
            df["Area"].nunique()
            if "Area" in df.columns
            else 0
        ),

        "unique_fields": (
            df["Field"].nunique()
            if "Field" in df.columns
            else 0
        ),

        "active_events": (
            df["IsActive"].sum()
            if "IsActive" in df.columns
            else 0
        ),

        "acked_events": (
            df["IsAcked"].sum()
            if "IsAcked" in df.columns
            else 0
        ),

        "normal_events": (
            df["IsNormal"].sum()
            if "IsNormal" in df.columns
            else 0
        ),
    }