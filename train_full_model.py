import pandas as pd
from btc_trend_classifier import train_trend_model
from btc_feature_engineering import create_features, label_trend

# Load full historical BTC dataset
df = pd.read_csv("btc_cleaning_till2025.csv")
df.columns = [col.lower().strip() for col in df.columns]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure 'price' column exists
if 'price' not in df.columns and 'close' in df.columns:
    df['price'] = df['close']

# --- Feature Engineering ---
df = label_trend(df)               # ✅ Add this line
df = create_features(df)

# --- Train model and save ---
model_full = train_trend_model(df, save=True, model_path="models/full_history_model.pkl")
