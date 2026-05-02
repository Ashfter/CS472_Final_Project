import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def load_unsw_nb15(
    file_path: str,
    sample_size: int = None,
    random_state: int = 42,
    top_n_services: int = 10,
):
    """
    Load and preprocess UNSW-NB15 for PCA + HDBSCAN.

    What this does:
    - optionally samples the dataset for faster runs / clearer graphs
    - keeps useful numeric + low-cardinality categorical features
    - cleans missing / invalid values
    - log-transforms heavy-tailed traffic columns
    - groups rare services into 'other'
    - one-hot encodes proto/service/state
    - removes constant columns

    Returns
    -------
    model_df : pd.DataFrame
        Preprocessed numeric feature matrix for PCA/HDBSCAN
    raw_df : pd.DataFrame
        Original sampled rows for reporting / evaluation
    meta : dict
        Small metadata bundle about what was kept
    """

    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # Optional sampling for speed / graph clarity
    if sample_size is not None and sample_size < len(df):
        if "label" in df.columns:
            # Stratified-ish sampling by label to keep both benign and attack rows
            sampled_parts = []
            total_len = len(df)

            for label_value, group in df.groupby("label"):
                target_n = max(1, int(round(sample_size * len(group) / total_len)))
                target_n = min(target_n, len(group))
                sampled_parts.append(
                    group.sample(n=target_n, random_state=random_state)
                )

            df = (
                pd.concat(sampled_parts, axis=0)
                .sample(frac=1.0, random_state=random_state)
                .reset_index(drop=True)
            )

            # If rounding made us overshoot sample_size, trim back down
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        else:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        print(f"Sampled {len(df)} rows from dataset")

    raw_df = df.copy()

    # Candidate columns commonly present in UNSW-NB15
    numeric_candidates = [
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "sloss",
        "dloss",
        "sinpkt",
        "dinpkt",
        "sjit",
        "djit",
        "swin",
        "stcpb",
        "dtcpb",
        "dwin",
        "tcprtt",
        "synack",
        "ackdat",
        "smean",
        "dmean",
        "trans_depth",
        "response_body_len",
        "ct_srv_src",
        "ct_state_ttl",
        "ct_dst_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
    ]

    categorical_candidates = [
        "proto",
        "service",
        "state",
    ]

    label_candidates = [
        "label",
        "attack_cat",
    ]

    present_numeric = [c for c in numeric_candidates if c in df.columns]
    present_categorical = [c for c in categorical_candidates if c in df.columns]
    present_labels = [c for c in label_candidates if c in df.columns]

    if not present_numeric and not present_categorical:
        raise ValueError(
            "No expected UNSW-NB15 columns found. Check that the CSV has headers."
        )

    model_df = df[present_numeric + present_categorical].copy()

    # Clean numeric columns
    for col in present_numeric:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric NaNs using median
    for col in present_numeric:
        median_value = model_df[col].median()
        if pd.isna(median_value):
            median_value = 0
        model_df[col] = model_df[col].fillna(median_value)

    # Log-transform heavy-tailed traffic features
    skewed_cols = [
        c for c in [
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "rate",
            "sload",
            "dload",
            "sloss",
            "dloss",
            "sinpkt",
            "dinpkt",
            "sjit",
            "djit",
            "trans_depth",
            "response_body_len",
            "ct_srv_src",
            "ct_state_ttl",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
        ]
        if c in model_df.columns
    ]

    for col in skewed_cols:
        model_df[col] = np.log1p(model_df[col].clip(lower=0))

    # Simple interpretable ratio features
    if "sbytes" in model_df.columns and "spkts" in model_df.columns:
        model_df["sbytes_per_pkt"] = model_df["sbytes"] / (model_df["spkts"] + 1)

    if "dbytes" in model_df.columns and "dpkts" in model_df.columns:
        model_df["dbytes_per_pkt"] = model_df["dbytes"] / (model_df["dpkts"] + 1)

    if "sbytes" in model_df.columns and "dbytes" in model_df.columns:
        model_df["byte_ratio_sd"] = model_df["sbytes"] / (model_df["dbytes"] + 1)

    if "spkts" in model_df.columns and "dpkts" in model_df.columns:
        model_df["pkt_ratio_sd"] = model_df["spkts"] / (model_df["dpkts"] + 1)

    # Clean categoricals
    for col in present_categorical:
        model_df[col] = (
            model_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "unknown", "": "unknown"})
        )

    # Collapse rare service values into 'other'
    if "service" in model_df.columns and top_n_services is not None and top_n_services > 0:
        top_services = model_df["service"].value_counts().nlargest(top_n_services).index
        model_df["service"] = model_df["service"].where(
            model_df["service"].isin(top_services),
            "other"
        )

    # One-hot encode low-cardinality protocol/state/service fields
    if present_categorical:
        model_df = pd.get_dummies(
            model_df,
            columns=present_categorical,
            dummy_na=False
        )

    # Final safety clean
    model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove constant columns
    if model_df.shape[1] > 0:
        selector = VarianceThreshold(threshold=0.0)
        selected_array = selector.fit_transform(model_df)

        kept_columns = model_df.columns[selector.get_support()]
        model_df = pd.DataFrame(
            selected_array,
            columns=kept_columns,
            index=model_df.index
        )

    meta = {
        "rows": len(model_df),
        "present_numeric": present_numeric,
        "present_categorical": present_categorical,
        "present_labels": present_labels,
        "final_feature_count": model_df.shape[1],
        "log_transformed_columns": skewed_cols,
    }

    print("\nUNSW-NB15 preprocessing summary")
    print("-------------------------------")
    print("Rows:", meta["rows"])
    print("Numeric columns kept:", len(meta["present_numeric"]))
    print("Categorical columns kept:", meta["present_categorical"])
    print("Label columns found:", meta["present_labels"])
    print("Final feature count after encoding/filtering:", meta["final_feature_count"])

    return model_df, raw_df, meta