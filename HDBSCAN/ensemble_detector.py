import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def run_isolation_forest(
    X: np.ndarray,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    predictions = clf.fit_predict(X)   # -1 = anomaly, 1 = normal
    scores = clf.decision_function(X)  # lower = more anomalous
    return predictions, scores


def combine_predictions(
    hdbscan_labels: np.ndarray,
    if_predictions: np.ndarray,
    mode: str = "union",
) -> np.ndarray:
    hdbscan_anomaly = hdbscan_labels == -1
    if_anomaly = if_predictions == -1
    if mode == "union":
        combined = hdbscan_anomaly | if_anomaly
    elif mode == "intersection":
        combined = hdbscan_anomaly & if_anomaly
    else:
        raise ValueError(f"mode must be 'union' or 'intersection', got {mode!r}")
    return np.where(combined, -1, 0)


def print_detector_comparison(
    raw_df: pd.DataFrame,
    hdbscan_labels: np.ndarray,
    if_predictions: np.ndarray,
    union_labels: np.ndarray,
    intersection_labels: np.ndarray,
) -> None:
    if "label" not in raw_df.columns:
        print("\nNo ground-truth 'label' column — skipping detector comparison.")
        return

    ground_truth = (raw_df["label"].values != 0).astype(int)

    def metrics(pred_anomaly: np.ndarray) -> tuple[float, float, float]:
        tp = int(((pred_anomaly == 1) & (ground_truth == 1)).sum())
        fp = int(((pred_anomaly == 1) & (ground_truth == 0)).sum())
        fn = int(((pred_anomaly == 0) & (ground_truth == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    detectors = [
        ("HDBSCAN alone",           (hdbscan_labels == -1).astype(int)),
        ("Isolation Forest alone",  (if_predictions == -1).astype(int)),
        ("Ensemble (union)",        (union_labels == -1).astype(int)),
        ("Ensemble (intersection)", (intersection_labels == -1).astype(int)),
    ]

    print("\nDetector comparison (vs ground truth)")
    print("-" * 55)
    print(f"{'Detector':<26} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("-" * 55)
    for name, pred in detectors:
        p, r, f = metrics(pred)
        print(f"{name:<26} {p:>9.4f} {r:>9.4f} {f:>9.4f}")
    print("-" * 55)


def plot_ensemble_comparison(
    X_2d: np.ndarray,
    hdbscan_labels: np.ndarray,
    if_predictions: np.ndarray,
    union_labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    label_sets = [hdbscan_labels, if_predictions, union_labels]
    titles = [
        f"HDBSCAN\n({(hdbscan_labels == -1).sum()} anomalies)",
        f"Isolation Forest\n({(if_predictions == -1).sum()} anomalies)",
        f"Ensemble (union)\n({(union_labels == -1).sum()} anomalies)",
    ]

    for ax, title, labels in zip(axes, titles, label_sets):
        anomaly = labels == -1
        ax.scatter(X_2d[~anomaly, 0], X_2d[~anomaly, 1], s=20, alpha=0.18, label="Normal")
        ax.scatter(X_2d[anomaly, 0], X_2d[anomaly, 1], s=40, alpha=0.9, label="Anomaly")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()

    plt.suptitle("Anomaly Detector Comparison", y=1.02)
    plt.tight_layout()
