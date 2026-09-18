import pandas as pd
from sklearn.decomposition import PCA


def create_pca_projection(
    scaled_features,
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a two-dimensional PCA projection for visualization.

    PCA is not used for HDBSCAN clustering.
    """

    if len(scaled_features) < 2:
        raise ValueError(
            "At least two observations are required "
            "for PCA visualization."
        )

    n_components = min(
        2,
        scaled_features.shape[1],
        scaled_features.shape[0],
    )

    pca = PCA(
        n_components=n_components
    )

    components = pca.fit_transform(
        scaled_features
    )

    result = window_df.copy()

    result["PCA1"] = components[:, 0]

    if n_components >= 2:
        result["PCA2"] = components[:, 1]
    else:
        result["PCA2"] = 0.0

    return result