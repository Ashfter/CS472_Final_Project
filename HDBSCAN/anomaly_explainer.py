import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = _safe_numeric(series).dropna()
    if clean.empty or pd.isna(value):
        return 50.0
    return float((clean <= value).mean() * 100.0)


def _median(series: pd.Series) -> float:
    clean = _safe_numeric(series).dropna()
    if clean.empty:
        return 0.0
    return float(clean.median())


def _ratio_to_median(series: pd.Series, value: float) -> float:
    med = _median(series)
    if med <= 0:
        return 1.0
    if pd.isna(value):
        return 1.0
    return float(value / med)


def _add_reason_from_percentile(
    reasons: list,
    reference_df: pd.DataFrame,
    row: pd.Series,
    column: str,
    high_text: str = None,
    low_text: str = None,
    high_cutoff: float = 99.0,
    low_cutoff: float = 1.0,
):
    if column not in row.index or column not in reference_df.columns:
        return

    value = pd.to_numeric(row[column], errors="coerce")
    p = _percentile_rank(reference_df[column], value)

    if high_text and p >= high_cutoff:
        reasons.append(high_text)
    elif low_text and p <= low_cutoff:
        reasons.append(low_text)


def _infer_behavior_tags(raw_row: pd.Series) -> list:
    """
    Pull simple behavior hints from raw categorical fields.
    These are hints, not labels.
    """
    tags = []

    proto = str(raw_row.get("proto", "")).strip().lower()
    service = str(raw_row.get("service", "")).strip().lower()
    state = str(raw_row.get("state", "")).strip().lower()

    if proto:
        tags.append(f"protocol={proto}")
    if service and service != "nan":
        tags.append(f"service={service}")
    if state and state != "nan":
        tags.append(f"state={state}")

    return tags


def describe_anomaly(
    processed_row: pd.Series,
    processed_reference_df: pd.DataFrame,
    raw_row: pd.Series = None,
    raw_reference_df: pd.DataFrame = None,
) -> str:
    """
    Create a human-readable anomaly description.

    Important design choice:
    - Use raw_df for human-facing magnitude reasoning when possible
    - Use processed_df only as a fallback
    """

    reasons = []
    tags = []

    if raw_row is None:
        raw_row = pd.Series(dtype=object)
    if raw_reference_df is None:
        raw_reference_df = pd.DataFrame()

    ref_df = raw_reference_df if not raw_reference_df.empty else processed_reference_df
    row = raw_row if not raw_row.empty else processed_row

    # Categorical hints
    tags.extend(_infer_behavior_tags(raw_row))

    # Volume / rate / duration clues
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "sbytes",
        high_text="extremely high source-byte volume",
        low_text="extremely low source-byte volume",
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "dbytes",
        high_text="extremely high destination-byte volume",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "rate",
        high_text="very high traffic rate",
        low_text="unusually low traffic rate",
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "dur",
        high_text="unusually long-lived connection",
        low_text="very short-lived connection",
    )

    # Packet profile clues
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "spkts",
        high_text="very high source packet count",
        low_text="very low source packet count",
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "dpkts",
        high_text="very high destination packet count",
        low_text="very low destination packet count",
    )

    # TTL clues
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "sttl",
        high_text="unusually high source TTL",
        low_text="unusually low source TTL",
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "dttl",
        high_text="unusually high destination TTL",
        low_text="unusually low destination TTL",
    )

    # Timing clues
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "tcprtt",
        high_text="abnormally high TCP round-trip timing",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "synack",
        high_text="slow SYN-ACK timing",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "ackdat",
        high_text="slow ACK/data timing",
        low_text=None,
    )

    # Spread / recurrence clues
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "ct_src_dport_ltm",
        high_text="many destination ports contacted recently (scan-like behavior)",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "ct_dst_ltm",
        high_text="repeated activity toward the same destination",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "ct_dst_src_ltm",
        high_text="unusually repetitive source-destination interaction",
        low_text=None,
    )
    _add_reason_from_percentile(
        reasons,
        ref_df,
        row,
        "ct_srv_src",
        high_text="high service reuse frequency from source",
        low_text=None,
    )

    # Higher-level behavior summaries using combinations
    # These are intentionally heuristic and descriptive.
    sbytes = pd.to_numeric(row.get("sbytes", np.nan), errors="coerce")
    dbytes = pd.to_numeric(row.get("dbytes", np.nan), errors="coerce")
    rate = pd.to_numeric(row.get("rate", np.nan), errors="coerce")
    dur = pd.to_numeric(row.get("dur", np.nan), errors="coerce")
    spkts = pd.to_numeric(row.get("spkts", np.nan), errors="coerce")
    dpkts = pd.to_numeric(row.get("dpkts", np.nan), errors="coerce")
    ct_src_dport_ltm = pd.to_numeric(row.get("ct_src_dport_ltm", np.nan), errors="coerce")

    if "sbytes" in ref_df.columns and "rate" in ref_df.columns:
        if (
            _percentile_rank(ref_df["sbytes"], sbytes) >= 99 and
            _percentile_rank(ref_df["rate"], rate) >= 99
        ):
            reasons.append("burst-like high-volume transfer behavior")

    if "dur" in ref_df.columns and "ct_src_dport_ltm" in ref_df.columns:
        if (
            _percentile_rank(ref_df["dur"], dur) <= 5 and
            _percentile_rank(ref_df["ct_src_dport_ltm"], ct_src_dport_ltm) >= 99
        ):
            reasons.append("short-duration multi-port behavior consistent with probing or scanning")

    if (
        "spkts" in ref_df.columns and
        "dpkts" in ref_df.columns and
        not pd.isna(spkts) and not pd.isna(dpkts)
    ):
        pkt_ratio = (spkts + 1) / (dpkts + 1)
        if pkt_ratio >= 10:
            reasons.append("strongly source-heavy packet imbalance")
        elif pkt_ratio <= 0.1:
            reasons.append("strongly destination-heavy packet imbalance")

    if (
        "sbytes" in ref_df.columns and
        "dbytes" in ref_df.columns and
        not pd.isna(sbytes) and not pd.isna(dbytes)
    ):
        byte_ratio = (sbytes + 1) / (dbytes + 1)
        if byte_ratio >= 10:
            reasons.append("strongly source-heavy byte imbalance")
        elif byte_ratio <= 0.1:
            reasons.append("strongly destination-heavy byte imbalance")

    # Remove duplicates while preserving order
    seen = set()
    unique_reasons = []
    for item in reasons:
        if item not in seen:
            unique_reasons.append(item)
            seen.add(item)

    seen = set()
    unique_tags = []
    for item in tags:
        if item not in seen:
            unique_tags.append(item)
            seen.add(item)

    if not unique_reasons:
        base = "isolated anomaly outside dense traffic clusters; no single dominant behavioral feature"
    else:
        base = "; ".join(unique_reasons)

    if unique_tags:
        return f"{base}. Context: " + ", ".join(unique_tags)

    return base


def _summarize_against_normal(
    processed_row: pd.Series,
    processed_reference_df: pd.DataFrame,
    labels: np.ndarray,
    current_index: int,
    top_k: int = 3,
) -> str:
    """
    Compare the anomaly to clustered points only (non-noise) using processed features.
    Returns a short summary of the strongest feature deviations.
    """
    if processed_reference_df.empty:
        return ""

    normal_mask = labels != -1
    if normal_mask.sum() == 0:
        return ""

    numeric_df = processed_reference_df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return ""

    row_numeric = pd.to_numeric(processed_row[numeric_df.columns], errors="coerce").fillna(0)

    normal_df = numeric_df.loc[normal_mask].copy()
    normal_median = normal_df.median()
    normal_std = normal_df.std(ddof=0).replace(0, np.nan)

    z = ((row_numeric - normal_median).abs() / normal_std).replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return ""

    top_features = z.sort_values(ascending=False).head(top_k).index.tolist()
    if not top_features:
        return ""

    pieces = []
    for feature in top_features:
        row_value = row_numeric[feature]
        median_value = normal_median[feature]
        direction = "higher" if row_value > median_value else "lower"
        pieces.append(f"{feature} is {direction} than clustered baseline")

    return "; ".join(pieces)


def build_anomaly_report(
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Build a human-readable anomaly report for all HDBSCAN noise points.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Processed feature matrix used for clustering
    raw_df : pd.DataFrame
        Original sampled UNSW rows for interpretation
    labels : np.ndarray
        HDBSCAN labels
    probabilities : np.ndarray
        HDBSCAN membership probabilities
    """

    feature_df = feature_df.copy().reset_index(drop=True)
    raw_df = raw_df.copy().reset_index(drop=True)

    noise_idx = np.where(labels == -1)[0]
    rows = []

    for idx in noise_idx:
        processed_row = feature_df.iloc[idx]
        raw_row = raw_df.iloc[idx] if idx < len(raw_df) else pd.Series(dtype=object)

        description = describe_anomaly(
            processed_row=processed_row,
            processed_reference_df=feature_df,
            raw_row=raw_row,
            raw_reference_df=raw_df,
        )

        baseline_comparison = _summarize_against_normal(
            processed_row=processed_row,
            processed_reference_df=feature_df,
            labels=labels,
            current_index=idx,
            top_k=3,
        )

        report_row = {
            "row_index": int(idx),
            "membership_probability": float(probabilities[idx]),
            "description": description,
            "baseline_comparison": baseline_comparison,
        }

        # Raw context fields
        for col in ["proto", "service", "state", "label", "attack_cat"]:
            if col in raw_df.columns:
                report_row[col] = raw_row[col]

        # Keep human-readable raw magnitudes when available
        for col in [
            "dur",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "rate",
            "sttl",
            "dttl",
            "ct_src_dport_ltm",
            "ct_dst_ltm",
            "ct_dst_src_ltm",
            "tcprtt",
            "synack",
            "ackdat",
        ]:
            if col in raw_df.columns:
                report_row[col] = raw_row[col]

        rows.append(report_row)

    report_df = pd.DataFrame(rows)

    if not report_df.empty:
        report_df = report_df.sort_values(
            by=["membership_probability", "row_index"],
            ascending=[True, True]
        ).reset_index(drop=True)

    return report_df