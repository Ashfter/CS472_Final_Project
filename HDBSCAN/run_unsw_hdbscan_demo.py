import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN

from unsw_nb15_loader import load_unsw_nb15
from PCA import run_pca
from anomaly_explainer import build_anomaly_report
from ensemble_detector import (
    run_isolation_forest,
    combine_predictions,
    print_detector_comparison,
    plot_ensemble_comparison,
)
from report_writer import write_html_report


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
    dataset_path = os.path.join("CSV_File_Creation", "UNSW_NB15_training-set(in).csv")

    model_df, raw_df, meta = load_unsw_nb15(
        dataset_path,
        sample_size=1000,
        random_state=42,
        top_n_services=10
    )

    os.makedirs("output", exist_ok=True)

    print("\nLoaded UNSW-NB15")
    print("Feature matrix shape:", model_df.shape)
    print("Raw data shape:", raw_df.shape)
    print("Metadata:", meta)

    pcs_df, pca_obj, scaler = run_pca(
        model_df,
        variance_threshold=0.95,
        scale=True
    )

    X = pcs_df.values

    clusterer = HDBSCAN(
        min_cluster_size=25,
        min_samples=8,
        metric="euclidean",
        cluster_selection_method="eom",
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

    # Calibrate IF contamination to the same anomaly rate HDBSCAN found
    estimated_contamination = max(0.01, min(0.5, float(n_noise / len(labels))))
    if_predictions, _ = run_isolation_forest(X, contamination=estimated_contamination)
    union_labels = combine_predictions(labels, if_predictions, mode="union")
    intersection_labels = combine_predictions(labels, if_predictions, mode="intersection")

    print("\nEnsemble summary")
    print("----------------")
    print("HDBSCAN anomalies:         ", n_noise)
    print("Isolation Forest anomalies:", int((if_predictions == -1).sum()))
    print("Union anomalies:           ", int((union_labels == -1).sum()))
    print("Intersection anomalies:    ", int((intersection_labels == -1).sum()))

    if X.shape[1] >= 2:
        X_2d = X[:, :2]
    else:
        X_2d = np.column_stack([X[:, 0], np.zeros(len(X))])

    plot_clusters(X_2d, labels)
    plot_confidence(X_2d, labels, probs)
    plot_ensemble_comparison(X_2d, labels, if_predictions, union_labels)

    print_basic_evaluation(raw_df, labels)
    print_detector_comparison(raw_df, labels, if_predictions, union_labels, intersection_labels)

    report_df = build_anomaly_report(
        feature_df=model_df,
        raw_df=raw_df,
        labels=labels,
        probabilities=probs
    )

    report_path = "output/anomaly_report.csv"
    if not report_df.empty:
        if_anomaly = if_predictions == -1
        report_df["if_flagged"] = [bool(if_anomaly[idx]) for idx in report_df["row_index"]]

    report_df.to_csv(report_path, index=False)
    print(f"\nSaved anomaly report to {report_path}")

    write_html_report(
        report_df=report_df,
        out_path="output/session_report.html",
        flows_analyzed=len(raw_df),
        session_label="UNSW-NB15 Demo",
    )

    plt.show()


if __name__ == "__main__":
    main()