import streamlit as st
import requests
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from btc_trend_classifier import load_google_sheet, label_trend, create_features, train_trend_model
from datetime import datetime, timedelta
from datetime import datetime
import gspread

def fetch_and_log_today_price(sheet, df):
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Skip if today's data already exists
        if today_str in df.index.strftime("%Y-%m-%d").tolist():
            return df  # ✅ Already logged

        # Fetch live BTC price
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url)
        price = float(response.json()['price'])

        # Calculate 24h and 7d changes
        df_sorted = df.sort_index()
        y_price = df_sorted['price'].iloc[-1] if len(df) >= 1 else price
        d7_price = df_sorted['price'].iloc[-7] if len(df) >= 7 else price

        change_24h = ((price - y_price) / y_price) * 100 if y_price else 0
        change_7d = ((price - d7_price) / d7_price) * 100 if d7_price else 0

        # Append to sheet
        new_row = [today_str, round(price, 2), round(change_24h, 2), round(change_7d, 2)]
        sheet.append_row(new_row)

        # Update df
        new_df = df.copy()
        new_df.loc[pd.to_datetime(today_str)] = {
            "price": price,
            "change_24h": change_24h,
            "change_7d": change_7d
        }
        return new_df

    except Exception as e:
        st.warning(f"⚠️ Could not log today's BTC data: {e}")
        return df

    except Exception as e:
        st.warning(f"⚠️ Could not log today's BTC data: {e}")
        return df

def get_live_btc_price_and_change():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = requests.get(url)
        data = response.json()
        price = float(data['lastPrice'])
        change_percent = float(data['priceChangePercent'])
        return price, change_percent
    except Exception as e:
        st.error(f"❌ Could not fetch live BTC price: {e}")
        return None, None



st.set_page_config(page_title="BTC Watchdog", layout="centered")


st.title("📈 BTC Trend Classifier")
live_price, change_percent = get_live_btc_price_and_change()
if live_price is not None:
    arrow = "🔺" if change_percent >= 0 else "🔻"
    st.metric(
        label="💸 Live BTC Price (Binance)",
        value=f"${live_price:,.2f}",
        delta=f"{arrow} {abs(change_percent):.2f}%"
    )



# --- Load data from Google Sheets ---
sheet_name = "btc_price_log"
worksheet_name = "DailyPrice"

# Re-authorize Google Sheet for write access
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
from oauth2client.service_account import ServiceAccountCredentials
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open(sheet_name).worksheet(worksheet_name)

# Load data and log today's BTC price
df = load_google_sheet(sheet_name, worksheet_name)
df = fetch_and_log_today_price(sheet, df)


try:
    # Load and log today's data if not yet added
    # Re-authorize for write access
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).worksheet(worksheet_name)

    # Load as DataFrame
    df = load_google_sheet(sheet_name, worksheet_name)

    # Log today if missing
    df = fetch_and_log_today_price(sheet, df)


    st.success("✅ Successfully loaded data from Google Sheet")
except Exception as e:
    st.error(f"❌ Failed to load Google Sheet: {e}")
    st.stop()


# --- Preprocess & Train ---
try:
    df = label_trend(df)
    df = create_features(df)
    st.info(f"🧮 Number of rows after feature engineering: {len(df)}")
    model = train_trend_model(df)
except Exception as e:
    st.error(f"❌ Error during model training: {e}")
    st.stop()

# --- Trend Prediction ---
latest_data = df.iloc[-1:][['ret_1d', 'ret_3d', 'ret_7d', 'volatility_7d']]
trend_prediction = model.predict(latest_data)[0]

st.metric("📊 Predicted Trend", trend_prediction)

# --- 7-Day BTC Price Trend ---
st.subheader("📈 7-Day BTC Price Trend")

# --- Your Personal History: Daily Log ---
st.subheader("🧾 Your Personal History: Daily Log")
try:
    st.dataframe(df[['price', 'change_24h', 'change_7d', 'trend']].tail(7))
except Exception as e:
    st.warning(f"⚠️ Could not show history table: {e}")


try:
    trend_df = df[-7:].copy()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(trend_df.index, trend_df['price'], marker='o')
    ax.set_title("BTC Price (Last 7 Days)")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=30)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"⚠️ Could not render 7-day trend chart: {e}")

# --- Change Summary ---
st.subheader("📉 Trend Summary (Last Entry)")
last_row = df.iloc[-1]

change_24h = last_row['change_24h']
change_7d = last_row['change_7d']

arrow_24h = "🔺" if change_24h > 0 else "🔻"
arrow_7d = "🔺" if change_7d > 0 else "🔻"

st.write(f"**24h Change:** {arrow_24h} {change_24h:.2f}%")
st.write(f"**7d Change:** {arrow_7d} {change_7d:.2f}%")


# --- Price Chart ---
st.subheader("📉 BTC Price (Last 30 Days)")
try:
    price_chart = df[-30:].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_chart.index, price_chart['price'], marker='o')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.set_title("BTC Price Over Time")
    ax.grid(True)

    # Format x-axis for clarity
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not render chart: {e}")

# --- Trend History Table ---
st.subheader("📘 BTC Trend History (Last 7 Days)")
trend_history = df[['price', 'trend']].tail(7).copy()
trend_history.index = trend_history.index.strftime('%Y-%m-%d')
st.table(trend_history)

# --- Bull Market Signal Detector ---
st.subheader("📈 Bull Market Signal Detector")

if len(df) < 8:
    st.warning("⚠️ Need at least 8 days of price data to detect bull signals (20%+ gain in 7 days).")
else:
    try:
        df['gain_7d'] = df['price'].pct_change(periods=7)
        bull_df = df[df['gain_7d'] >= 0.2]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['price'], label="BTC Price", color="blue")

        if not bull_df.empty:
            ax.scatter(bull_df.index, bull_df['price'], color="red", label="20%+ Gain", zorder=5)

        ax.set_title("BTC Price with Bull Signals (20%+ Gain in 7 Days)")
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.xticks(rotation=45)

        st.pyplot(fig)

        if bull_df.empty:
            st.info("ℹ️ No 20%+ bull signals detected yet.")
    except Exception as e:
        st.warning(f"⚠️ Could not run signal detector: {e}")


# --- Raw Data Preview ---
with st.expander("📄 View raw data"):
    st.dataframe(df.tail(10))
