import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hdbscan

from HDBSCAN.unsw_nb15_loader import load_unsw_nb15
from HDBSCAN.PCA import run_pca
from HDBSCAN.anomaly_explainer import build_anomaly_report


def plot_clusters(X_2d: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        alpha = 0.35 if label == -1 else 0.85
        label_name = "Noise (-1)" if label == -1 else f"Cluster {label}"
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=25, alpha=alpha, label=label_name)

    plt.title("HDBSCAN Clusters on UNSW-NB15 (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()


def plot_confidence(X_2d: np.ndarray, labels: np.ndarray, probs: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=np.clip(probs * 120, 10, 120),
        alpha=np.clip(probs, 0.2, 1.0)
    )
    plt.title("HDBSCAN Membership Confidence")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()


def plot_noise(X_2d: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(10, 6))
    normal_mask = labels != -1
    noise_mask = labels == -1

    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], s=20, alpha=0.18, label="Clustered")
    plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], s=35, alpha=0.95, label="Anomaly candidates")
    plt.title("Noise / Anomaly Candidates")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()


def print_basic_evaluation(raw_df: pd.DataFrame, labels: np.ndarray):
    if "label" not in raw_df.columns:
        print("\nNo ground-truth 'label' column found, skipping evaluation.")
        return

    eval_df = raw_df.copy()
    eval_df["cluster"] = labels
    eval_df["pred_anomaly"] = (eval_df["cluster"] == -1).astype(int)

    tp = int(((eval_df["pred_anomaly"] == 1) & (eval_df["label"] == 1)).sum())
    fp = int(((eval_df["pred_anomaly"] == 1) & (eval_df["label"] == 0)).sum())
    tn = int(((eval_df["pred_anomaly"] == 0) & (eval_df["label"] == 0)).sum())
    fn = int(((eval_df["pred_anomaly"] == 0) & (eval_df["label"] == 1)).sum())

    print("\nBasic anomaly evaluation")
    print("------------------------")
    print("TP:", tp)
    print("FP:", fp)
    print("TN:", tn)
    print("FN:", fn)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    if "attack_cat" in eval_df.columns:
        noise_attacks = (
            eval_df[eval_df["pred_anomaly"] == 1]["attack_cat"]
            .fillna("Unknown")
            .value_counts()
        )
        print("\nAttack categories among anomaly candidates:")
        print(noise_attacks.head(10))


def main():
    # Change this path to your dataset path
    dataset_path = os.path.join("CSV_File_Creatation", "UNSW_NB15_training-set(in).csv")

    model_df, raw_df, meta = load_unsw_nb15(
        dataset_path,
        sample_size=1000,
        random_state=42,
        top_n_services=10
    )

    # Save a small sample of the processed feature matrix for presentation
    sample_rows = 25
    preprocessed_sample = model_df.head(sample_rows).copy()
    preprocessed_sample.to_csv("preprocessed_sample.csv", index=False)

    print("\nSaved preprocessed sample to preprocessed_sample.csv")
    print(preprocessed_sample.to_string(index=False))

    print("Loaded UNSW-NB15")
    print("Feature matrix shape:", model_df.shape)
    print("Raw data shape:", raw_df.shape)
    print("Metadata:", meta)

    pcs_df, pca_obj, scaler = run_pca(
        model_df,
        variance_threshold=0.95,
        scale=True
    )

    X = pcs_df.values

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    labels = clusterer.fit_predict(X)
    probs = clusterer.probabilities_

    unique_labels = np.unique(labels)
    n_clusters = len([x for x in unique_labels if x != -1])
    n_noise = int((labels == -1).sum())

    print("\nHDBSCAN summary")
    print("---------------")
    print("Unique labels:", unique_labels)
    print("Number of clusters:", n_clusters)
    print("Number of noise points:", n_noise)

    if X.shape[1] >= 2:
        X_2d = X[:, :2]
    else:
        X_2d = np.column_stack([X[:, 0], np.zeros(len(X))])

    plot_clusters(X_2d, labels)
    plot_confidence(X_2d, labels, probs)
    plot_noise(X_2d, labels)

    print_basic_evaluation(raw_df, labels)

    report_df = build_anomaly_report(
        feature_df=model_df,
        raw_df=raw_df,
        labels=labels,
        probabilities=probs
    )

    report_path = "anomaly_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved anomaly report to {report_path}")

    if not report_df.empty:
        preview_cols = [
            c for c in [
                "row_index",
                "membership_probability",
                "proto",
                "service",
                "state",
                "label",
                "attack_cat",
                "description",
                "baseline_comparison",
            ]
            if c in report_df.columns
        ]
        print("\nTop anomaly examples:")
        print(report_df[preview_cols].head(10).to_string(index=False))

    plt.show()


if __name__ == "__main__":
    main()