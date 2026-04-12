import pandas as pd
import numpy as np


def get_numeric_network_data(file_path, save_file=None, return_df=True):
    """
    Load either:
    - large datasets like UNSW-NB15 (many columns), or
    - custom 9-column baseline CSVs with no header

    Returns strictly numeric network features for PCA/HDBSCAN.

    Output columns:
    ['sport', 'dsport', 'proto', 'dur', 'sbytes']
    """

    # Peek at the first row to infer shape
    sample = pd.read_csv(file_path, nrows=1, header=None)
    col_count = sample.shape[1]

    if col_count >= 40:
        # Likely UNSW-NB15-like layout
        # pull only the fields we want:
        # srcip, sport, dstip, dsport, proto, dur, sbytes
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3, 4, 6, 7])
        df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'dur', 'sbytes']
    elif col_count >= 9:
        # Custom baseline layout with no header:
        # srcip, sport, dstip, dsport, proto, dur, sbytes, attack_cat, label
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[:, :7]
        df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'dur', 'sbytes']
    else:
        raise ValueError(
            f"Unsupported CSV format in {file_path}. Expected 9-column custom baseline or wider UNSW-like format."
        )

    # Normalize protocol strings -> numeric values
    proto_map = {
        'tcp': 6,
        'udp': 17,
        'icmp': 1
    }

    df['proto'] = (
        df['proto']
        .astype(str)
        .str.strip()
        .str.lower()
        .map(proto_map)
        .fillna(0)
    )

    # Force numeric conversion
    for col in ['sport', 'dsport', 'dur', 'sbytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    numeric_df = df[['sport', 'dsport', 'proto', 'dur', 'sbytes']].copy()

    if save_file:
        numeric_df.to_csv(save_file, header=False, index=False)
        print(f"Saved numeric data to {save_file}")

    if return_df:
        return numeric_df

    return numeric_df.values