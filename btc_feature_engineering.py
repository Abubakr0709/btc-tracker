import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    df['ret_1d'] = df['price'].pct_change(1)
    df['ret_3d'] = df['price'].pct_change(3)
    df['ret_7d'] = df['price'].pct_change(7)
    df['volatility_7d'] = df['price'].pct_change().rolling(7).std()
    df.dropna(inplace=True)
    return df

def label_trend(df):
    df = df.copy()
    df['target'] = (df['price'].shift(-7) > df['price']).astype(int)
    df.dropna(inplace=True)
    return df
