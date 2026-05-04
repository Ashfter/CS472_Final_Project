from __future__ import annotations

import math
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    if v < 1024:
        return f"{v:.0f} B"
    if v < 1024 ** 2:
        return f"{v / 1024:.1f} KB"
    return f"{v / 1024 ** 2:.1f} MB"


def _fmt_num(val, unit: str = "", decimals: int = 2) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    s = f"{v:,.{decimals}f}"
    return f"{s} {unit}".strip() if unit else s


# ---------------------------------------------------------------------------
# Attack inference
# ---------------------------------------------------------------------------

_ATTACK_PATTERNS: list[tuple[str, list[str], int]] = [
    ("Port Scan / Reconnaissance", [
        "multi-port", "scan", "probing", "many destination ports",
    ], 2),
    ("DDoS / Traffic Flood", [
        "very high traffic rate", "burst-like high-volume",
        "source-heavy packet imbalance", "very high source packet count",
    ], 2),
    ("Data Exfiltration", [
        "extremely high source-byte volume", "source-heavy byte imbalance",
        "burst-like high-volume",
    ], 2),
    ("Brute Force / Credential Stuffing", [
        "repeated activity toward the same destination",
        "unusually repetitive source-destination",
        "high service reuse frequency",
    ], 2),
    ("C2 Beaconing", [
        "unusually repetitive source-destination",
        "repeated activity toward the same destination",
        "unusually long-lived connection",
    ], 1),
    ("Slow Loris / Connection Exhaustion", [
        "abnormally high tcp round-trip",
        "slow syn-ack timing",
        "slow ack/data timing",
        "unusually long-lived connection",
    ], 2),
]

_ATTACK_DESCRIPTIONS: dict[str, str] = {
    "Port Scan / Reconnaissance": (
        "A host appears to be probing multiple ports in rapid succession. "
        "This is commonly the first stage of an attack — mapping what services are exposed."
    ),
    "DDoS / Traffic Flood": (
        "Traffic volume and rate are abnormally high. "
        "This pattern is consistent with an attempt to overwhelm a service or consume bandwidth."
    ),
    "Data Exfiltration": (
        "A large volume of data is being sent outbound with little incoming traffic. "
        "This could indicate sensitive data being transferred out of the network."
    ),
    "Brute Force / Credential Stuffing": (
        "Repeated connections to the same destination service suggest automated login attempts "
        "or credential testing against an exposed service."
    ),
    "C2 Beaconing": (
        "Persistent, repetitive contact with the same external host may indicate a compromised "
        "device checking in with a command-and-control server."
    ),
    "Slow Loris / Connection Exhaustion": (
        "TCP handshake and acknowledgment timings are abnormally slow. This can indicate a "
        "Slow Loris-style attack designed to hold connections open and exhaust server resources."
    ),
    "Unknown / Behavioral Anomaly": (
        "This flow does not match a specific known attack signature, but its traffic pattern "
        "is statistically unusual compared to normal session behavior."
    ),
}


def _infer_attack(row: pd.Series, description: str) -> tuple[str, str, list[str]]:
    """Returns (attack_type, likelihood, signals)."""
    desc_lower = description.lower()
    if_confirmed = bool(row.get("if_flagged", False))

    votes: dict[str, int] = {}
    signals: list[str] = []

    for attack_type, keywords, weight in _ATTACK_PATTERNS:
        for kw in keywords:
            if kw in desc_lower:
                votes[attack_type] = votes.get(attack_type, 0) + weight
                signal = kw.capitalize()
                if signal not in signals:
                    signals.append(signal)

    if if_confirmed:
        for k in votes:
            votes[k] += 1

    if not votes:
        return "Unknown / Behavioral Anomaly", "Low", [
            "No specific attack signature matched",
            "Flow is statistically isolated from normal traffic clusters",
        ]

    attack_type = max(votes, key=lambda k: votes[k])
    top_score = votes[attack_type]

    if top_score >= 5 or (top_score >= 4 and if_confirmed):
        likelihood = "High"
    elif top_score >= 3 or if_confirmed:
        likelihood = "Medium"
    else:
        likelihood = "Low"

    if if_confirmed:
        signals.append("Confirmed by both detectors independently")

    return attack_type, likelihood, signals


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _finding_html(i: int, row: pd.Series) -> str:
    description = str(row.get("description", "")).strip() or \
        "Isolated anomaly — no single dominant behavioral feature."
    attack_type, likelihood, signals = _infer_attack(row, description)
    attack_desc = _ATTACK_DESCRIPTIONS.get(attack_type, "")

    proto   = str(row.get("proto",   "")).strip()
    service = str(row.get("service", "")).strip()
    state   = str(row.get("state",   "")).strip()
    context_parts = [p for p in [proto, service, state]
                     if p and p not in ("nan", "unknown", "")]
    context_line = " / ".join(context_parts) or "unknown"

    detection = "Both detectors" if row.get("if_flagged", False) else "Single detector"

    # Traffic stats table rows
    stat_rows = ""
    for label, key, fmt in [
        ("Duration",  "dur",    lambda v: _fmt_num(v, "s", 3)),
        ("Sent",      "sbytes", _fmt_bytes),
        ("Received",  "dbytes", _fmt_bytes),
        ("Rate",      "rate",   lambda v: _fmt_num(v, "pkt/s", 1)),
        ("Src pkts",  "spkts",  lambda v: _fmt_num(v, "", 0)),
        ("Dst pkts",  "dpkts",  lambda v: _fmt_num(v, "", 0)),
    ]:
        if key in row.index:
            stat_rows += f"<tr><td>{label}</td><td>{fmt(row[key])}</td></tr>"

    stats_html = (
        f'<table class="stats-table">{stat_rows}</table>'
        if stat_rows else ""
    )

    baseline = str(row.get("baseline_comparison", "")).strip()
    baseline_html = (
        f'<p class="field"><span class="field-label">Deviation from normal:</span> {baseline}</p>'
        if baseline and baseline != "nan" else ""
    )

    signals_text = "; ".join(signals) if signals else "none"

    return f"""<div class="finding">
  <div class="finding-header">
    [{i}] {attack_type} &mdash; Likelihood: {likelihood} &mdash; {context_line} &mdash; {detection}
  </div>
  <p class="attack-desc">{attack_desc}</p>
  {stats_html}
  <p class="field"><span class="field-label">Signals:</span> {signals_text}</p>
  <p class="field"><span class="field-label">Behavior:</span> {description[0].upper() + description[1:]}</p>
  {baseline_html}
</div>"""


def write_html_report(
    report_df: pd.DataFrame,
    out_path: str,
    flows_analyzed: int,
    session_label: str = "Network Session",
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")

    n_anomalies = len(report_df)
    n_normal = flows_analyzed - n_anomalies
    n_high_conf = int(report_df["if_flagged"].sum()) if "if_flagged" in report_df.columns else 0

    likelihood_counts: dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}
    if not report_df.empty:
        for _, row in report_df.iterrows():
            desc = str(row.get("description", ""))
            _, lh, _ = _infer_attack(row, desc)
            likelihood_counts[lh] += 1

    if report_df.empty:
        findings_html = "<p>No anomalies detected this session.</p>"
    else:
        findings_html = "\n".join(
            _finding_html(i, row)
            for i, (_, row) in enumerate(report_df.iterrows(), 1)
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Network Sentinel &mdash; {session_label}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.6;
      color: #111;
      background: #fff;
      padding: 32px 48px;
    }}

    h1 {{ font-size: 18px; font-weight: 700; margin-bottom: 2px; }}
    .subtitle {{ color: #555; font-size: 13px; margin-bottom: 24px; }}

    hr {{ border: none; border-top: 1px solid #ccc; margin: 20px 0; }}

    /* Summary row */
    .summary-row {{
      display: flex;
      gap: 0;
      margin-bottom: 20px;
      border-top: 1px solid #ccc;
      border-bottom: 1px solid #ccc;
      padding: 14px 0;
    }}
    .summary-item {{
      flex: 1;
      text-align: center;
      border-right: 1px solid #ddd;
      padding: 0 12px;
    }}
    .summary-item:first-child {{ text-align: left; padding-left: 0; }}
    .summary-item:last-child {{ border-right: none; }}
    .summary-item .s-num {{ font-size: 22px; font-weight: 700; }}
    .summary-item .s-label {{ font-size: 11px; color: #777; margin-top: 1px; }}

    /* Section heading */
    h2 {{ font-size: 14px; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.5px; color: #333; margin-bottom: 14px; }}

    /* Individual finding */
    .finding {{
      border-left: 3px solid #999;
      padding: 12px 0 12px 16px;
      margin-bottom: 20px;
    }}
    .finding-header {{
      font-weight: 600;
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .attack-desc {{
      font-size: 13px;
      color: #444;
      margin-bottom: 10px;
    }}
    .stats-table {{
      border-collapse: collapse;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .stats-table td {{
      padding: 1px 20px 1px 0;
      color: #333;
    }}
    .stats-table td:first-child {{ color: #777; }}
    .field {{
      font-size: 12px;
      color: #444;
      margin-top: 4px;
    }}
    .field-label {{ color: #777; }}

    .footer {{ margin-top: 32px; font-size: 11px; color: #aaa; }}
  </style>
</head>
<body>

  <h1>Network Sentinel &mdash; Session Report</h1>
  <div class="subtitle">{session_label} &nbsp;&middot;&nbsp; {timestamp}</div>

  <div class="summary-row">
    <div class="summary-item"><div class="s-num">{flows_analyzed:,}</div><div class="s-label">Flows Analyzed</div></div>
    <div class="summary-item"><div class="s-num">{n_normal:,}</div><div class="s-label">Normal</div></div>
    <div class="summary-item"><div class="s-num">{n_anomalies}</div><div class="s-label">Anomalies</div></div>
    <div class="summary-item"><div class="s-num">{n_high_conf}</div><div class="s-label">Both Detectors</div></div>
    <div class="summary-item"><div class="s-num">{likelihood_counts["High"]}</div><div class="s-label">High Likelihood</div></div>
    <div class="summary-item"><div class="s-num">{likelihood_counts["Medium"]}</div><div class="s-label">Medium Likelihood</div></div>
    <div class="summary-item"><div class="s-num">{likelihood_counts["Low"]}</div><div class="s-label">Low / Unknown</div></div>
  </div>

  <hr>

  <h2>Findings</h2>
  {findings_html}

  <div class="footer">
    Generated by Network Sentinel &nbsp;&middot;&nbsp; HDBSCAN + Isolation Forest ensemble
  </div>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved session report to {out_path}")
