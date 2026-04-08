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
) -> Tuple[pd.DataFrame, SKPCA, Optional[StandardScaler]]:

    # Load data if path provided
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found for PCA.")

    # Scale features if requested
    scaler: Optional[StandardScaler]
    if scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(numeric_df.values)
    else:
        scaler = None
        features = numeric_df.values

    # Decide number of components
    max_components = min(features.shape[0], features.shape[1])
    if n_components is None:
        # compute PCA with full components to derive explained ratio
        pca_tmp = SKPCA(n_components=max_components)
        pca_tmp.fit(features)
        cum_var = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_components = max(1, min(n_components, max_components))

    pca = SKPCA(n_components=n_components)
    transformed = pca.fit_transform(features)

    # Build output DataFrame
    pc_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=pc_names, index=numeric_df.index)

    if save_path:
        transformed_df.to_csv(save_path, index=True)

    return transformed_df, pca, scaler


if __name__ == "__main__":
    # Quick local smoke test when running directly
    try:
        sample = pd.DataFrame(np.random.randn(100, 6), columns=[f"f{i}" for i in range(6)])
        pcs, pca_obj, scaler_obj = run_pca(sample, variance_threshold=0.9)
        print(pcs.head())
    except Exception as e:
        print("PCA smoke test failed:", e)
