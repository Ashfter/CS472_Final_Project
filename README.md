# Network Sentinel
### HDBSCAN + Isolation Forest Anomaly Detection for Home Networks
*CS472 — Unsupervised Machine Learning Final Project*

---

## Overview

Network Sentinel detects anomalous traffic on a local network using an ensemble of two unsupervised machine learning algorithms. It requires no labeled data and no prior knowledge of attack signatures — it learns what normal traffic looks like and flags statistical outliers.

At the end of a session, it produces a plain HTML report identifying each anomaly, inferring the most likely attack type, and rating the likelihood based on observed behavioral signals.

---

## How It Works

**1. Traffic Capture**
`logCreator.py` uses Scapy to sniff packets on the local interface and aggregates them into network flows (grouped by source IP, destination IP, port, and protocol). Baseline captures for specific applications (Minecraft, Terraria, Monster Hunter Wild) are stored in `CSV_File_Creation/`.

**2. Preprocessing**
The preprocessing pipeline (`unsw_nb15_loader.py`) prepares raw flow data for clustering:
- Numeric features are cleaned and NaN-filled with column medians
- Heavy-tailed traffic columns (`sbytes`, `rate`, `dpkts`, etc.) are log-transformed to compress outlier scale
- Ratio features are engineered (`sbytes_per_pkt`, `byte_ratio_sd`, etc.)
- Categorical fields (`proto`, `service`, `state`) are one-hot encoded
- Constant columns are removed via variance thresholding

**3. Dimensionality Reduction**
`PCA.py` applies StandardScaler followed by PCA, retaining enough components to explain 95% of variance. This reduces ~50 features to ~8 principal components, improving clustering density and speed.

**4. Ensemble Detection**
Two independent detectors run on the same PCA-reduced data:
- **HDBSCAN** — density-based clustering; points too isolated to belong to any cluster are flagged as noise (anomalies)
- **Isolation Forest** — tree-based; anomalies are points that require fewer splits to isolate

Their results are combined into a union (flagged by either) and an intersection (flagged by both). Points confirmed by both detectors are treated as high-confidence anomalies.

**5. Reporting**
Each anomaly is analyzed by `anomaly_explainer.py`, which compares its traffic features against the normal population using percentile ranking and z-scores to produce a plain-English behavioral description. `report_writer.py` then infers the most likely attack type (port scan, DDoS, data exfiltration, etc.) and generates an HTML session report.

---

## Project Structure

```
CS472_Final_Project/
├── requirements.txt
├── CSV_File_Creation/
│   ├── baseline_Minecraft.csv
│   ├── baseline_Terraria.csv
│   └── baseline_MH_Wild.csv
└── HDBSCAN/
    ├── logCreator.py            # Live packet capture → baseline CSV
    ├── databaseFormatter.py     # Loads baseline CSVs for clustering
    ├── unsw_nb15_loader.py      # Loads and preprocesses UNSW-NB15
    ├── PCA.py                   # Dimensionality reduction
    ├── ensemble_detector.py     # Isolation Forest + ensemble logic
    ├── anomaly_explainer.py     # Behavioral description generation
    ├── report_writer.py         # HTML session report
    ├── run_hdbscan.py           # Pipeline entry point for baseline CSVs
    └── run_unsw_hdbscan_demo.py # Pipeline entry point for UNSW-NB15
```

---

## Setup

```
pip install -r requirements.txt
```

Requires scikit-learn >= 1.3. The `hdbscan` package is not used — `sklearn.cluster.HDBSCAN` is used instead and requires no C compiler.

**UNSW-NB15 dataset** is not included in this repository due to size. Download it from [the UNSW research page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place the files in `CSV_File_Creation/`:
- `UNSW_NB15_training-set(in).csv`
- `UNSW-NB15_4.csv`

---

## Usage

**Run the full pipeline on UNSW-NB15 (evaluation mode):**
```
python -m HDBSCAN.run_unsw_hdbscan_demo
```
Outputs written to `output/`:
- `session_report.html` — human-readable anomaly report
- `anomaly_report.csv` — raw anomaly data

**Run on local baseline captures:**
```
python -m HDBSCAN.run_hdbscan
```

**Capture a new application baseline (requires admin/root for packet sniffing):**
```
python -m HDBSCAN.logCreator
```

---

## Evaluation Dataset

Benchmarking uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), a network intrusion dataset containing both normal traffic and labeled attack categories (Fuzzers, DoS, Reconnaissance, Exploits, etc.). Ground-truth labels are used only for evaluating detector precision and recall — the detection algorithms themselves receive no label information.

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| HDBSCAN alone | — | — | — |
| Isolation Forest alone | — | — | — |
| Ensemble (union) | — | — | — |
| Ensemble (intersection) | — | — | — |

*Fill in after running `run_unsw_hdbscan_demo.py`.*

---

## Session Report

The HTML report (`output/session_report.html`) opens in any browser. For each detected anomaly it shows:

- **Inferred attack type** — Port Scan, DDoS, Data Exfiltration, Brute Force, C2 Beaconing, Slow Loris, or Unknown
- **Likelihood rating** — High / Medium / Low, based on how many behavioral signals fired and whether both detectors agreed
- **Traffic stats** — duration, bytes sent/received, packet counts, rate
- **Behavioral description** — plain-English explanation of what made this flow statistically unusual
- **Deviation from normal** — which specific features deviated most from clustered baseline traffic
