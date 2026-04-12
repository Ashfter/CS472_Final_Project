import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import hdbscan
except Exception as e:
    raise RuntimeError(
        "The 'hdbscan' package is not installed. Install it with:\n"
        "pip install hdbscan"
    ) from e

from HDBSCAN.databaseFormatter import get_numeric_network_data
from HDBSCAN.PCA import run_pca


def find_csv_dir(project_root: str) -> str:
    candidates = [
        os.path.join(project_root, "CSV_Creation"),
        os.path.join(project_root, "CSV_File_Creation"),
        os.path.join(project_root, "CSV_File_Creatation"),
        os.path.join(project_root, "CSV_Creation/"),
        os.path.join(project_root, "CSV_File_Creation/"),
        os.path.join(project_root, "CSV_File_Creatation/"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find a CSV directory. Checked: "
        "CSV_Creation, CSV_File_Creation, CSV_File_Creatation"
    )


def load_all_csvs(csv_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames = []

    for path in csv_paths:
        try:
            df = get_numeric_network_data(path, return_df=True)
            frames.append(df)
            print(f"Loaded {os.path.basename(path)} -> shape {df.shape}")
        except Exception as e:
            print(f"Skipping {os.path.basename(path)}: {e}")

    if not frames:
        raise RuntimeError("No usable CSV data was loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # Drop duplicate rows and obvious garbage if desired
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\nCombined dataset shape: {combined.shape}")
    return combined


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                s=25, alpha=0.35, label="Noise (-1)"
            )
        else:
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                s=30, alpha=0.85, label=f"Cluster {label}"
            )

    plt.title("HDBSCAN Clusters in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()


def plot_membership_confidence(X_2d: np.ndarray, labels: np.ndarray, probabilities: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=np.clip(probabilities * 120, 10, 140),
        alpha=np.clip(probabilities, 0.2, 1.0)
    )
    plt.title("HDBSCAN Membership Confidence")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()


def plot_noise_highlight(X_2d: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(10, 6))

    clustered_mask = labels != -1
    noise_mask = labels == -1

    plt.scatter(
        X_2d[clustered_mask, 0],
        X_2d[clustered_mask, 1],
        s=20,
        alpha=0.18,
        label="Clustered / Normal-ish"
    )
    plt.scatter(
        X_2d[noise_mask, 0],
        X_2d[noise_mask, 1],
        s=40,
        alpha=0.9,
        label="Noise / Potential Anomaly"
    )

    plt.title("Potential Anomalies Highlighted")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()


def main():
    # Assumes this file lives in HDBSCAN/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = find_csv_dir(project_root)

    numeric_df = load_all_csvs(csv_dir)

    # PCA for clustering and plotting
    pcs_df, pca_obj, scaler = run_pca(
        numeric_df,
        variance_threshold=0.95,
        scale=True
    )

    X = pcs_df.values

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    labels = clusterer.fit_predict(X)
    probabilities = clusterer.probabilities_

    unique_labels = np.unique(labels)
    num_clusters = len([x for x in unique_labels if x != -1])
    num_noise = int(np.sum(labels == -1))

    print("\nHDBSCAN results")
    print("----------------")
    print("Unique labels:", unique_labels)
    print("Cluster count:", num_clusters)
    print("Noise points:", num_noise)

    # For graphs, use first two PCs if available
    if X.shape[1] >= 2:
        X_2d = X[:, :2]
    else:
        X_2d = np.column_stack([X[:, 0], np.zeros(X.shape[0])])

    plot_clusters_2d(X_2d, labels)
    plot_membership_confidence(X_2d, labels, probabilities)
    plot_noise_highlight(X_2d, labels)

    plt.show()


if __name__ == "__main__":
    main()