import pandas as pd

# --- Bull Cycle Detection ---
def detect_bull_cycles(df, threshold=0.2, window=7):
    df = df.copy()
    df['gain_7d'] = df['price'].pct_change(periods=window)
    df['bull_signal'] = df['gain_7d'] >= threshold
    return df

# --- Cluster Bull Runs (group consecutive bull days) ---
def cluster_bull_signals(df):
    df = df.copy()
    df['bull_cluster'] = 0
    cluster_id = 0
    in_cluster = False

    for i in range(len(df)):
        if df.iloc[i]['bull_signal']:
            if not in_cluster:
                cluster_id += 1
                in_cluster = True
            df.iat[i, df.columns.get_loc('bull_cluster')] = cluster_id
        else:
            in_cluster = False

    return df[df['bull_cluster'] > 0]

# --- Normalize current cycle vs past ---
def align_cycles(df, peak_dates):
    """
    Aligns multiple bull runs by day-count from start of cluster.
    peak_dates = list of pd.Timestamp for each known bull start
    """
    aligned = {}
    for date in peak_dates:
        start_idx = df.index.get_loc(date)
        segment = df.iloc[start_idx:].copy()
        segment = segment.reset_index()
        segment['day'] = range(len(segment))
        aligned[date.strftime('%Y-%m-%d')] = segment[['day', 'price']]
    return aligned
