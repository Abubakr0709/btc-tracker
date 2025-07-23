# btc_sheet_logger.py
import gspread
import requests
import pandas as pd
import json
import os
from datetime import date
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

# --- Load Google Credentials ---

import os
import json
from google.oauth2.service_account import Credentials


try:
    creds_data = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])  # Render uses this
except KeyError:
    with open("credentials.json") as f:  # Fallback for local dev
        creds_data = json.load(f)


scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_data, scopes=scope)

client = gspread.authorize(creds)
sheet = client.open("btc_price_log").worksheet("DailyPrice")  # Make sure this matches

# --- Fetch live BTC price and 24h change from CoinGecko ---
def get_live_price():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&market_data=true"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = round(data["market_data"]["current_price"]["usd"], 2)
        change_24h = round(data["market_data"]["price_change_percentage_24h"], 2)
        return price, change_24h
    except Exception as e:
        print(f"[ERROR] Failed to fetch live BTC price: {e}")
        return None, None

# --- Log today's data ---
today = date.today().isoformat()
btc_price, change_24h = get_live_price()

if btc_price is None:
    print("⚠️ Price unavailable. Skipping logging.")
    exit()

# Load existing sheet records
data = sheet.get_all_records()
df_sheet = pd.DataFrame(data)

for col in ['price', 'change_24h', 'change_7d']:
    if col in df_sheet.columns:
        df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')


# Prevent duplicates
if df_sheet.empty or "Date" not in df_sheet.columns:
    sheet.append_row([today, btc_price, change_24h, "", ""])
    print("✅ BTC data logged (first entry).")
elif today not in df_sheet["Date"].astype(str).values:
    sheet.append_row([today, btc_price, change_24h, "", ""])
    print("✅ BTC data logged to Google Sheets.")
else:
    print("🟡 Already logged today.")

# --- Optional: Append to local CSV log ---
csv_path = "btc_price_log.csv"
new_entry = pd.DataFrame([[today, btc_price, change_24h]], columns=["date", "price", "change_24h"])

if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    df_csv.columns = [col.lower().strip() for col in df_csv.columns]  # <- Normalize column names

    if "date" not in df_csv.columns:
        print("❌ 'date' column not found in CSV. Please check the file format.")
    elif today not in df_csv["date"].astype(str).values:
        df_csv = pd.concat([df_csv, new_entry], ignore_index=True)
        df_csv.to_csv(csv_path, index=False)
        print("✅ Logged to local CSV as well.")
    else:
        print("🟡 Already logged to local CSV.")

else:
    new_entry.to_csv(csv_path, index=False)
    print("✅ Created local CSV log.")
