import requests
import pandas as pd
import time
from datetime import datetime

def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_time=None, end_time=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    data = response.json()
    return data

def get_all_binance_data():
    all_data = []
    start = int(datetime(2017, 8, 1).timestamp() * 1000)  # Binance BTC/USDT starts in Aug 2017
    now = int(time.time() * 1000)

    while start < now:
        print(f"Fetching data from {datetime.fromtimestamp(start/1000)}")
        batch = fetch_binance_klines(start_time=start)
        if not batch:
            break
        all_data.extend(batch)
        last_timestamp = batch[-1][0]
        start = last_timestamp + 1
        time.sleep(1)  # avoid rate limit

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"
    ])

    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df["CloseTime"] = pd.to_datetime(df["CloseTime"], unit="ms")
    df.set_index("OpenTime", inplace=True)

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.to_csv("btc_binance_daily.csv")
    print("✅ BTC daily data saved to btc_binance_daily.csv")

if __name__ == "__main__":
    get_all_binance_data()
