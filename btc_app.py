import streamlit as st
import requests
import pandas as pd
from btc_trend_classifier import load_google_sheet, label_trend, create_features, train_trend_model
from datetime import datetime
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import json
import random
import plotly.graph_objects as go

# --- Streamlit Setup ---
st.set_page_config(page_title="🐺 BTC Watchdog", layout="centered", initial_sidebar_state="auto")
st.title("🐺 BTC Watchdog Dashboard")

# --- Live Price Fetch ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_live_btc_price_and_change():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url)
        data = response.json()

        if "bitcoin" not in data or "usd" not in data["bitcoin"] or "usd_24h_change" not in data["bitcoin"]:
            raise ValueError("CoinGecko API returned incomplete data")

        price = float(data["bitcoin"]["usd"])
        change_percent = float(data["bitcoin"]["usd_24h_change"])

        return price, change_percent

    except Exception as e:
        st.error(f"❌ Could not fetch live BTC price (CoinGecko): {e}")
        return None, None

# --- Google Sheets Setup ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_google_sheet(sheet_name, worksheet_name):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

        credentials_json = st.secrets["google_credentials"]["value"]
        credentials_dict = json.loads(credentials_json)

        creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        client = gspread.authorize(creds)
        sheet = client.open(sheet_name).worksheet(worksheet_name)

        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        df.columns = [col.lower().strip() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return df  # Return the DataFrame

    except Exception as e:
        st.error(f"❌ Failed to load Google Sheet: {e}")
        return None

# --- Fetch & Log Today's Price ---
def fetch_and_log_today_price(df):
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if 'price' not in df.columns:
            df['price'] = pd.Series(dtype='float64')

        if today_str in df.index.strftime("%Y-%m-%d").tolist():
            return df

        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url)
        price = float(response.json()['price'])

        df_sorted = df.sort_index()
        y_price = df_sorted['price'].iloc[-1] if len(df) >= 1 else price
        d7_price = df_sorted['price'].iloc[-7] if len(df) >= 7 else price

        change_24h = ((price - y_price) / y_price) * 100 if y_price else 0
        change_7d = ((price - d7_price) / d7_price) * 100 if d7_price else 0

        new_row = [today_str, round(price, 2), round(change_24h, 2), round(change_7d, 2)]
        df.loc[pd.to_datetime(today_str)] = {
            "price": price,
            "change_24h": change_24h,
            "change_7d": change_7d
        }

        return df

    except Exception as e:
        st.warning(f"⚠️ Could not log today's BTC data: {e}")
        return df

# --- Altseason Score (Placeholder function) ---
def altseason_score(df):
    score = 27.91  # Example score
    level = "LOW"
    msg = "📉 BTC still dominant — altseason not ready"
    mock_dominance = 43.61
    mock_eth_ratio = 0.0634
    return score, level, msg, mock_dominance, mock_eth_ratio

# --- App Start ---
live_price, change_percent = get_live_btc_price_and_change()

if live_price is not None and change_percent is not None:
    arrow = "🔺" if change_percent >= 0 else "🔻"
    st.metric("💸 Live BTC Price (CoinGecko)", f"${live_price:,.2f}", f"{arrow} {abs(change_percent):.2f}%")
else:
    st.error("❌ Could not fetch BTC price data.")

try:
    # Load Google Sheet and DataFrame
    sheet_name = "btc_price_log"
    worksheet_name = "DailyPrice"
    df = load_google_sheet(sheet_name, worksheet_name)

    if df is not None:
        # Fetch and log today's price
        df = fetch_and_log_today_price(df)
        st.success("✅ Successfully loaded data from Google Sheet")

        # Check if 'price' column exists in df
        if 'price' not in df.columns:
            st.error("❌ 'price' column is missing from the data.")
            st.stop()

        # Process the DataFrame
        df = label_trend(df)
        df = create_features(df)

        st.info(f"🧮 Rows after feature engineering: {len(df)}")

        # Train the model
        model = train_trend_model(df)

    else:
        st.error("❌ Failed to load data from Google Sheet.")

except Exception as e:
    st.error(f"❌ Failed to load Google Sheet: {e}")
    st.stop()

# --- Trend Prediction ---
latest_data = df.iloc[-1:][['ret_1d', 'ret_3d', 'ret_7d', 'volatility_7d']]
trend_prediction = model.predict(latest_data)[0]
st.metric("📊 Predicted Trend", trend_prediction)


# --- Altseason Score ---
st.subheader("🔀 Altseason Rotation Signal")

try:
    score, level, msg, mock_dominance, mock_eth_ratio = altseason_score(df)
    st.metric("🧠 Altseason Score", f"{score}%", f"{level}")
    st.write(msg)
    st.caption(f"Simulated BTC Dominance: {mock_dominance:.2f}%")
    st.caption(f"Simulated ETH/BTC Ratio: {mock_eth_ratio:.4f}")

except Exception as e:
    st.warning(f"⚠️ Could not calculate Altseason Score: {e}")

# --- 7-Day BTC Price Trend ---
st.subheader("📈 7-Day BTC Price Trend")
try:
    trend_df = df[-7:].copy()  # Ensure you are taking the last 7 days of data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df.index, y=trend_df['price'], mode='lines+markers', name='BTC Price'))
    fig.update_layout(template="plotly_dark", title="BTC Price (Last 7 Days)", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Could not render 7-day trend chart: {e}")


# --- Personal History Log ---
st.subheader("🧾 Your Personal History: Daily Log")
try:
    history_df = df[['price', 'change_24h', 'change_7d', 'trend']].tail(7).copy()
    history_df = history_df.reset_index(drop=True)  # Reset index to avoid non-unique index issues

    for col in ['price', 'change_24h', 'change_7d']:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')

    styled_df = history_df
    styled_df = history_df.style \
        .format("{:.2f}", subset=['price', 'change_24h', 'change_7d']) \
        .background_gradient(subset=['change_24h'], cmap='Greens') \
        .background_gradient(subset=['change_7d'], cmap='Blues') \
        .set_properties(**{'background-color': '#111', 'color': 'white'})

    st.dataframe(styled_df, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Could not show history table: {e}")

# --- Change Summary ---
st.subheader("📉 Trend Summary (Last Entry)")
last_row = df.iloc[-1]
change_24h = last_row['change_24h']
change_7d = last_row['change_7d']
arrow_24h = "🔺" if change_24h > 0 else "🔻"
arrow_7d = "🔺" if change_7d > 0 else "🔻"
st.write(f"**24h Change:** {arrow_24h} {change_24h:.2f}%")
st.write(f"**7d Change:** {arrow_7d} {change_7d:.2f}%")

# --- 30-Day Price Chart ---
st.subheader("📉 BTC Price (Last 30 Days)")
try:
    price_chart = df[-30:].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_chart.index, y=price_chart['price'], mode='lines+markers', name='BTC Price'))
    fig.update_layout(template="plotly_dark", title="BTC Price Over Last 30 Days", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Could not render 30-day trend chart: {e}")

# --- Trend History Table ---
st.subheader("📘 BTC Trend History (Last 7 Days)")
try:
    # Extract the last 7 days of price and trend data
    trend_history = df[['price', 'trend']].tail(7).copy()

    # Reset the index to ensure it's unique
    trend_history = trend_history.reset_index(drop=True)  # Resetting the index

    # Ensure 'price' is numeric
    trend_history['price'] = pd.to_numeric(trend_history['price'], errors='coerce')

    # Apply styling
    styled_trend = trend_history.style \
        .format("{:.2f}", subset=['price']) \
        .set_properties(**{'background-color': '#111', 'color': 'white'})

    st.dataframe(styled_trend, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not render trend history table: {e}")

# --- Bull Market Signal Detector ---
st.subheader("📈 Bull Market Signal Detector")
if len(df) < 8:
    st.warning("⚠️ Need at least 8 days of price data to detect bull signals (20%+ gain in 7 days).")
else:
    try:
        df['gain_7d'] = df['price'].pct_change(periods=7)
        bull_df = df[df['gain_7d'] >= 0.2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='BTC Price'))
        if not bull_df.empty:
            fig.add_trace(go.Scatter(x=bull_df.index, y=bull_df['price'], mode='markers', marker=dict(color='red', size=10), name='20%+ Gain'))
        fig.update_layout(template="plotly_dark", title="BTC Price with Bull Signals (20%+ Gain in 7 Days)", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        if bull_df.empty:
            st.info("ℹ️ No 20%+ bull signals detected yet.")
    except Exception as e:
        st.warning(f"⚠️ Could not run signal detector: {e}")


# --- Raw Data Preview ---
with st.expander("📄 View raw data"):
    st.dataframe(df.tail(10))


# --- Change Summary ---
st.subheader("📉 Trend Summary (Last Entry)")
last_row = df.iloc[-1]
change_24h = last_row['change_24h']
change_7d = last_row['change_7d']
arrow_24h = "🔺" if change_24h > 0 else "🔻"
arrow_7d = "🔺" if change_7d > 0 else "🔻"
st.write(f"**24h Change:** {arrow_24h} {change_24h:.2f}%")
st.write(f"**7d Change:** {arrow_7d} {change_7d:.2f}%")

# --- 30-Day Price Chart ---
st.subheader("📉 BTC Price (Last 30 Days)")
try:
    price_chart = df[-30:].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_chart.index, y=price_chart['price'], mode='lines+markers', name='BTC Price'))
    fig.update_layout(template="plotly_dark", title="BTC Price Over Last 30 Days", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Could not render 30-day trend chart: {e}")

# --- Trend History Table ---
st.subheader("📘 BTC Trend History (Last 7 Days)")
try:
    trend_history = df[['price', 'trend']].tail(7).copy()
    trend_history.index = trend_history.index.strftime('%Y-%m-%d')
    trend_history['price'] = pd.to_numeric(trend_history['price'], errors='coerce')

    styled_trend = trend_history.style \
        .format("{:.2f}", subset=['price']) \
        .set_properties(**{'background-color': '#111', 'color': 'white'})

    st.dataframe(styled_trend, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Could not render trend history table: {e}")

# --- Bull Market Signal Detector ---
st.subheader("📈 Bull Market Signal Detector")
if len(df) < 8:
    st.warning("⚠️ Need at least 8 days of price data to detect bull signals (20%+ gain in 7 days).")
else:
    try:
        df['gain_7d'] = df['price'].pct_change(periods=7)
        bull_df = df[df['gain_7d'] >= 0.2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='BTC Price'))
        if not bull_df.empty:
            fig.add_trace(go.Scatter(x=bull_df.index, y=bull_df['price'], mode='markers', marker=dict(color='red', size=10), name='20%+ Gain'))
        fig.update_layout(template="plotly_dark", title="BTC Price with Bull Signals (20%+ Gain in 7 Days)", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        if bull_df.empty:
            st.info("ℹ️ No 20%+ bull signals detected yet.")
    except Exception as e:
        st.warning(f"⚠️ Could not run signal detector: {e}")

# --- Raw Data Preview ---
with st.expander("📄 View raw data"):
    st.dataframe(df.tail(10))
