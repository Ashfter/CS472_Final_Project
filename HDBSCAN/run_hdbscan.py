import os
import sys
import glob
import importlib
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import hdbscan
except Exception:
    hdbscan = None


def find_csv_dir(root: str) -> str:
    # Prefer the actual folder present in the repo; accept common misspelling
    candidates = [
        os.path.join(root, "CSV_File_Creatation"),
        os.path.join(root, "CSV_File_Creation"),
        os.path.join(root, "CSV_File_Creatation/"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not find CSV_File_Creatation or CSV_File_Creation directory")


def load_processors(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    db_mod = importlib.import_module("HDBSCAN.databaseFormater")
    pca_mod = importlib.import_module("HDBSCAN.PCA")

    db_proc = getattr(db_mod, "process", None)
    pca_proc = getattr(pca_mod, "process", None)

    if not callable(db_proc):
        raise ImportError("`databaseFormater` does not expose a callable `process(df)` function")
    if not callable(pca_proc):
        raise ImportError("`PCA` does not expose a callable `process(df)` function")

    return db_proc, pca_proc


def process_file(path: str, db_proc, pca_proc) -> np.ndarray:
    df = pd.read_csv(path)
    df_db = db_proc(df)
    transformed = pca_proc(df_db)
    # Accept either DataFrame or numpy array
    if isinstance(transformed, pd.DataFrame):
        return transformed.values
    if isinstance(transformed, np.ndarray):
        return transformed
    # Try to coerce
    return np.asarray(transformed)


def main():
    # use repository root (parent of this HDBSCAN package) so CSV files
    # stored at the repo root are found correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = find_csv_dir(project_root)

    db_proc, pca_proc = load_processors(project_root)

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {csv_dir}")
        return

    transformed_list: List[np.ndarray] = []
    for p in csv_paths:
        try:
            arr = process_file(p, db_proc, pca_proc)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            transformed_list.append(arr)
            print(f"Processed {os.path.basename(p)} -> shape {arr.shape}")
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not transformed_list:
        print("No transformed data available to cluster")
        return

    X = np.vstack(transformed_list)
    print(f"Aggregated data shape: {X.shape}")

    if hdbscan is None:
        raise RuntimeError("hdbscan package not available. Install it with `pip install hdbscan` to run clustering")

    clusterer = hdbscan.HDBSCAN()
    labels = clusterer.fit_predict(X)

    unique_labels = np.unique(labels)
    print(f"HDBSCAN ran successfully. Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters (noise label -1).")

    # Visualizations
    def _project_2d(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] >= 2:
            return arr[:, :2]
        try:
            from sklearn.decomposition import PCA as SKPCA
            return SKPCA(n_components=2).fit_transform(arr)
        except Exception:
            # pad with zeros if only 1 feature available
            return np.hstack([arr, np.zeros((arr.shape[0], 1))])

    proj = _project_2d(X)

    probs = getattr(clusterer, 'probabilities_', None)
    if probs is None:
        probs = np.ones_like(labels, dtype=float)
        probs[labels == -1] = 0.35

    sns.set(style="whitegrid")

    # Scatter by cluster label
    plt.figure(figsize=(8, 6))
    unique = np.unique(labels)
    palette = sns.color_palette('tab20', n_colors=max(2, len(unique)))
    for i, lab in enumerate(unique):
        mask = labels == lab
        color = 'lightgray' if lab == -1 else palette[i % len(palette)]
        plt.scatter(proj[mask, 0], proj[mask, 1], s=40, c=[color], label=f"{lab}", alpha=0.9 if lab != -1 else 0.4, edgecolors='k', linewidths=0.2)
    plt.title('HDBSCAN clusters (2D projection)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='label', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Scatter with membership probabilities as alpha/size
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab20', s=np.clip(probs * 80, 10, 200), alpha=np.clip(probs, 0.2, 1.0), edgecolors='k', linewidths=0.2)
    plt.colorbar(sc, label='cluster label')
    plt.title('Points sized/alpha by membership probability')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Optional: condensed tree if plotting utilities available
    try:
        from hdbscan import plots as hdbplots
        try:
            fig = hdbplots.plot_condensed_tree(clusterer)
            fig.suptitle('HDBSCAN condensed tree')
        except Exception:
            # some versions return an Axes object
            hdbplots.plot_condensed_tree(clusterer)
    except Exception:
        print('hdbscan plotting utilities not available; skipping condensed tree.')

    plt.show()


if __name__ == "__main__":
    main()
