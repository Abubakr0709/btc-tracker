import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date
import os

# Load merged BTC dataset
df_all = pd.read_csv("btc_all_time.csv")
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all.set_index('Date', inplace=True)

@st.cache_data(ttl=300)
def get_historical_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7"
    response = requests.get(url)
    return response.json()

# Page config
st.set_page_config(page_title="BTC Tracker", layout="centered")
st.title("📈 Bitcoin Live Tracker")

# ========== LIVE PRICE SECTION ==========

# Get current price + 24h change from CoinGecko
live_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
response = requests.get(live_url)
data = response.json()

if 'bitcoin' in data:
    btc_price = data['bitcoin']['usd']
    btc_change_24h = data['bitcoin']['usd_24h_change']
    st.metric(label="💰 BTC Price (USD)", value=f"${btc_price:,.2f}", delta=f"{btc_change_24h:.2f}% (24h)")
else:
    st.error("⚠️ Failed to fetch live BTC price. Please try again later.")

# ========== HISTORICAL TREND CHART ==========

st.subheader("📊 7-Day BTC Price Trend")

try:
    hist_data = get_historical_data()

    if 'prices' in hist_data:
        prices = hist_data['prices']
        df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)

        # Calculate 7-day % change
        price_start = df['Price'].iloc[0]
        price_end = df['Price'].iloc[-1]
        change_7d = ((price_end - price_start) / price_start) * 100

        # ========= STEP 3: LOG PRICE TO CSV FILE =========
        csv_file = "btc_price_log.csv"
        today = date.today().isoformat()
        log_entry = {
            "Date": today,
            "Price": btc_price,
            "Change_24h": btc_change_24h,
            "Change_7d": change_7d
        }

        if os.path.exists(csv_file):
            df_log = pd.read_csv(csv_file)
            if today not in df_log["Date"].values:
                df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
                df_log.to_csv(csv_file, index=False)
        else:
            df_log = pd.DataFrame([log_entry])
            df_log.to_csv(csv_file, index=False)

        # Display metrics and chart
        st.metric(label="📆 7-Day Change", value=f"{change_7d:.2f}%", delta=None)
        st.line_chart(df['Price'])

    else:
        st.error("⚠️ API returned no price data.")
        st.write("Raw API response:", hist_data)

except Exception as e:
    st.error(f"⚠️ Error loading chart data: {e}")

# ========== STEP 4: SHOW LOGGED HISTORY ==========

st.subheader("🗂️ BTC Daily Log (Your Personal History)")

# Check if the file exists
if os.path.exists("btc_price_log.csv"):
    history_df = pd.read_csv("btc_price_log.csv")

    # Show full table
    st.dataframe(history_df.sort_values("Date", ascending=False), use_container_width=True)

    # Line chart of BTC Price
    st.line_chart(history_df.set_index("Date")["Price"])
else:
    st.info("No BTC log found yet — run the app for a few days to build your dataset.")

# ========== STEP 5: Detect Bull Market Peak Signal ==========

st.subheader("🚨 Bull Market Signal Detector")

if os.path.exists("btc_price_log.csv"):
    history_df = pd.read_csv("btc_price_log.csv")

    if len(history_df) >= 7:
        # Compare the most recent 7 days
        recent = history_df.tail(7)

        # Calculate total 7-day gain
        start_price = recent.iloc[0]["Price"]
        end_price = recent.iloc[-1]["Price"]
        total_gain = ((end_price - start_price) / start_price) * 100
        latest_24h = recent.iloc[-1]["Change_24h"]

        if total_gain > 20 and latest_24h < 1:
            st.error(f"⚠️ Warning: BTC has gained {total_gain:.2f}% in the last 7 days, but 24h gain is only {latest_24h:.2f}%. This setup looks like previous bull market slowdowns.")
        else:
            st.success("✅ No signs of a peak — current price action is healthy.")
    else:
        st.info("Need at least 7 days of data to detect peak signals.")
        
# ========== DAILY GOOGLE SHEET LOGGING ==========

from datetime import date
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def log_to_google_sheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        sheet = client.open("btc_price_log").sheet1
        data = sheet.get_all_records()
        df_sheet = pd.DataFrame(data)

        today = date.today().isoformat()

        if today not in df_sheet["Date"].values:
            new_row = [today, btc_price, btc_change_24h, change_7d]
            sheet.append_row(new_row)
            st.success("✅ BTC data logged to Google Sheet.")
        else:
            st.info("🟡 Already logged today.")
    except Exception as e:
        st.error(f"❌ Google Sheet logging failed: {e}")

log_to_google_sheet()
