import pandas as pd
from btc_feature_engineering import create_features, label_trend
from btc_trend_classifier import train_trend_model
from datetime import datetime
import os

# Load latest data from Google Sheet CSV backup (already automated)
df = pd.read_csv("btc_price_log.csv")  # or fetch live if you're syncing directly
df.columns = [col.lower().strip() for col in df.columns]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure price column exists
if 'price' not in df.columns and 'close' in df.columns:
    df['price'] = df['close']

# Label and create features
df = label_trend(df)
df = create_features(df)

# Train and overwrite the model
model = train_trend_model(df, save=True, model_path="models/latest_model.pkl")

# Update timestamp
with open("models/last_trained.txt", "w") as f:
    f.write(datetime.now().isoformat())

print("✅ Daily model retrained and saved.")
