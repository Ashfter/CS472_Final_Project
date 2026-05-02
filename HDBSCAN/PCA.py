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
) -> Tuple[pd.DataFrame, SKPCA, Optional[StandardScaler]]:
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found for PCA.")

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df.values)
    else:
        scaler = None
        X = numeric_df.values

    max_components = min(X.shape[0], X.shape[1])

    if n_components is None:
        pca_tmp = SKPCA(n_components=max_components)
        pca_tmp.fit(X)
        cum_var = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_components = max(1, min(n_components, max_components))

    pca = SKPCA(n_components=n_components)
    transformed = pca.fit_transform(X)

    cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, columns=cols, index=numeric_df.index)

    print("\nPCA summary")
    print("-----------")
    print("Input shape:", numeric_df.shape)
    print("Output shape:", transformed_df.shape)
    print("Components:", n_components)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

    return transformed_df, pca, scaler