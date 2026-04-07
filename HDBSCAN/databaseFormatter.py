import pandas as pd
import numpy as np

def get_numeric_network_data(file_path, save_file=None):
    """
    Returns a headerless NumPy array of strictly numbers for PCA/ML.
    Drops IPs and converts Protocol to numeric values.
    """
    # 1. Detect and Load (Logic from before)
    sample = pd.read_csv(file_path, nrows=1, header=None)
    col_count = sample.shape[1]
    
    if col_count >= 40:
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3, 4, 6, 7])
    elif col_count == 9:
        df = pd.read_csv(file_path, header=None).iloc[:, :7]
    else:
        df = pd.read_csv(file_path, header=None)

    df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'dur', 'sbytes']

    # 2. CONVERT TO NUMBERS
    # Convert Protocol to numbers (TCP=6, UDP=17, others=0)
    proto_map = {'tcp': 6, 'udp': 17}
    df['proto'] = df['proto'].str.lower().map(proto_map).fillna(0)

    # Convert ports to numbers (handling non-numeric noise)
    df['sport'] = pd.to_numeric(df['sport'], errors='coerce').fillna(0)
    df['dsport'] = pd.to_numeric(df['dsport'], errors='coerce').fillna(0)

    # 3. DROP STRINGS (IP Addresses)
    # PCA cannot use IP addresses. We keep: sport, dsport, proto, dur, sbytes
    numeric_df = df[['sport', 'dsport', 'proto', 'dur', 'sbytes']]

    # 4. EXPORT / RETURN
    if save_file:
        # Save with NO header and NO index (strictly numbers)
        numeric_df.to_csv(save_file, header=False, index=False)
        print(f"Saved strictly numeric data to {save_file}")

    # Return as a matrix (NumPy array) for your PCA script
    return numeric_df.values

# --- How to use it in your workflow ---
# data_for_pca = get_numeric_network_data("baseline_valorant.csv", "ready_for_pca.csv")

# Now 'data_for_pca' is a clean mathematical matrix:
# [[63942, 1900, 17, 3.034, 848], 
#  [...]]