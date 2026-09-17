import pandas as pd

from sklearn.decomposition import PCA


def create_pca_projection(
    scaled_features,
    window_df: pd.DataFrame,
) -> pd.DataFrame:

    pca = PCA(
        n_components=2
    )

    components = pca.fit_transform(
        scaled_features
    )

    result = window_df.copy()

    result["PCA1"] = components[:, 0]
    result["PCA2"] = components[:, 1]

    return result