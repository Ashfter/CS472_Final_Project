from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    data: Union[str, pd.DataFrame],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    scale: bool = True,
    save_path: Optional[str] = None,
    headerless_csv: bool = True,
) -> Tuple[pd.DataFrame, SKPCA, Optional[StandardScaler]]:
    """
    Run PCA on numeric network data.

    Parameters
    ----------
    data : str or DataFrame
        Either a CSV path or a DataFrame.
    n_components : int or None
        If None, chooses the smallest number of components meeting variance_threshold.
    variance_threshold : float
        Cumulative explained variance target.
    scale : bool
        Standardize features before PCA.
    save_path : str or None
        Optional CSV output path.
    headerless_csv : bool
        When data is a path, read without headers by default.

    Returns
    -------
    transformed_df, pca, scaler
    """

    if isinstance(data, str):
        if headerless_csv:
            df = pd.read_csv(data, header=None)
        else:
            df = pd.read_csv(data)
    else:
        df = data.copy()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found for PCA.")

    if scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(numeric_df.values)
    else:
        scaler = None
        features = numeric_df.values

    max_components = min(features.shape[0], features.shape[1])

    if n_components is None:
        pca_tmp = SKPCA(n_components=max_components)
        pca_tmp.fit(features)
        cum_var = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_components = max(1, min(n_components, max_components))

    pca = SKPCA(n_components=n_components)
    transformed = pca.fit_transform(features)

    pc_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=pc_names, index=numeric_df.index)

    print(f"\nPCA input shape: {numeric_df.shape}")
    print(f"PCA output shape: {transformed_df.shape}")
    print(f"Chosen components: {n_components}")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

    if save_path:
        transformed_df.to_csv(save_path, index=False)
        print(f"Saved PCA output to {save_path}")

    return transformed_df, pca, scaler


if __name__ == "__main__":
    try:
        sample = pd.DataFrame(
            np.random.randn(100, 5),
            columns=["sport", "dsport", "proto", "dur", "sbytes"]
        )
        pcs, pca_obj, scaler_obj = run_pca(sample, variance_threshold=0.95)
        print(pcs.head())
    except Exception as e:
        print("PCA smoke test failed:", e)