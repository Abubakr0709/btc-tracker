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
from btc_cycle_analyzer import detect_bull_cycles, cluster_bull_signals, align_cycles


# --- Load Full Historical BTC Data ---
@st.cache_data(ttl=86400)
def load_full_btc_data(csv_path="btc_cleaning_till2025.csv"):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.lower().strip() for col in df.columns]

        # Rename 'close' to 'price' if needed
        if "close" in df.columns:
            df.rename(columns={"close": "price"}, inplace=True)

        if "price" not in df.columns:
            raise KeyError("❌ No 'price' column found.")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return df
    except Exception as e:
        st.error(f"❌ Failed to load full BTC history: {e}")
        return None



# --- Streamlit Setup ---
st.set_page_config(page_title="🐺 BTC Watchdog", layout="centered", initial_sidebar_state="auto")
st.title("🐺 BTC Watchdog Dashboard")

# --- Live Price Fetch ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_live_btc_price_and_change():
    try:
        # Current price
        url_current = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response_current = requests.get(url_current)
        price_now = float(response_current.json()['price'])

        # Simulated 24h ago price using 24hr ticker data
        url_24h = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response_24h = requests.get(url_24h)
        data_24h = response_24h.json()

        price_24h_ago = price_now / (1 + float(data_24h["priceChangePercent"]) / 100)
        change_percent = ((price_now - price_24h_ago) / price_24h_ago) * 100

        return price_now, change_percent

    except Exception as e:
        st.error(f"❌ Could not fetch live BTC price (Binance): {e}")
        return None, None
    
def get_altseason_metrics():
    try:
        # ETH/BTC and USD prices
        url_price = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd,btc"
        price_data = requests.get(url_price).json()

        eth_btc = price_data["ethereum"]["btc"]
        btc_usd = price_data["bitcoin"]["usd"]
        eth_usd = price_data["ethereum"]["usd"]

        # BTC dominance
        url_dom = "https://api.coingecko.com/api/v3/global"
        dom_data = requests.get(url_dom).json()
        btc_dominance = dom_data["data"]["market_cap_percentage"]["btc"]

        return btc_dominance, eth_btc, btc_usd, eth_usd
    except Exception as e:
        st.warning(f"⚠️ Could not fetch altseason metrics: {e}")
        return None, None, None, None


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
    
# --- Load Full Historical BTC Data (2010–2025) ---
@st.cache_data(ttl=86400)
def load_full_btc_data(csv_path="btc_cleaning_till2025.csv"):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.lower().strip() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # ✅ Ensure 'price' column exists
        if "price" not in df.columns and "close" in df.columns:
            df["price"] = df["close"]

        return df
    except Exception as e:
        st.error(f"❌ Failed to load full BTC history: {e}")
        return None

btc_full_df = load_full_btc_data()
# --- Process Full BTC History for Trend Modeling ---
if btc_full_df is not None:
    from btc_trend_classifier import label_trend, create_features

    btc_full_df = label_trend(btc_full_df)
    btc_full_df = create_features(btc_full_df)

    st.info(f"📊 Processed full BTC dataset: {len(btc_full_df)} rows after feature engineering")
    # --- Train Model on Full History ---
    from btc_trend_classifier import train_trend_model

    st.subheader("🧠 Training on Full BTC History (2010–2025)")

    model_full = train_trend_model(btc_full_df)

    st.success("✅ Model trained on full historical BTC data!")


if btc_full_df is not None:
    st.success(f"📚 Loaded full BTC history: {len(btc_full_df)} rows")
    st.dataframe(btc_full_df.tail(3))  # Optional preview

    # --- Bull Cluster Shaded Chart ---
    st.subheader("📈 Historical Bull Market Clusters")

    try:
        import matplotlib.pyplot as plt

        weekly = btc_full_df['price'].resample('W').last()
        weekly_change = weekly.pct_change()
        bull_mask = weekly_change >= 0.20

        bull_runs = []
        run_start = None

        for i in range(len(bull_mask)):
            if bull_mask[i] and run_start is None:
                run_start = bull_mask.index[i]
            elif not bull_mask[i] and run_start is not None:
                run_end = bull_mask.index[i - 1]
                bull_runs.append((run_start, run_end))
                run_start = None
        if run_start is not None:
            bull_runs.append((run_start, bull_mask.index[-1]))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(weekly.index, weekly, label="BTC Weekly Close", color='blue', linewidth=1.2)

        for start, end in bull_runs:
            ax.axvspan(start, end, color='green', alpha=0.3)

        ax.set_title("BTC Bull Market Clusters (2010–2025)")
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Could not render bull cluster chart: {e}")

# --- Bull Cluster Summary Table ---
st.subheader("📘 Bull Run Periods Summary Table")


try:
    # Reuse weekly and bull_mask from previous block
    bull_weeks = weekly_change[weekly_change >= 0.20]

    # Identify consecutive bull run periods
    bull_runs = []
    run_start = None

    for i in range(len(bull_mask)):
        if bull_mask[i] and run_start is None:
            run_start = bull_mask.index[i]
        elif not bull_mask[i] and run_start is not None:
            run_end = bull_mask.index[i - 1]
            if run_start != run_end:
                duration = (run_end - run_start).days // 7
                gain = (weekly[run_end] - weekly[run_start]) / weekly[run_start] * 100
                bull_runs.append((run_start.date(), run_end.date(), duration, round(gain, 2)))
            run_start = None

    # Handle last run if still open
    if run_start is not None and run_start != bull_mask.index[-1]:
        run_end = bull_mask.index[-1]
        duration = (run_end - run_start).days // 7
        gain = (weekly[run_end] - weekly[run_start]) / weekly[run_start] * 100
        bull_runs.append((run_start.date(), run_end.date(), duration, round(gain, 2)))


    summary_df = pd.DataFrame(bull_runs, columns=["Start", "End", "Duration (weeks)", "Total Gain (%)"])
    st.dataframe(summary_df.style.format({"Total Gain (%)": "{:.2f}"}))


    # Optional: Add yearly count
    st.subheader("📊 Bull Run Frequency by Year")
    summary_df["Year"] = pd.to_datetime(summary_df["Start"]).dt.year
    yearly_counts = summary_df["Year"].value_counts().sort_index()
    st.bar_chart(yearly_counts)

except Exception as e:
    st.error(f"❌ Could not generate bull summary: {e}")



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
    st.metric("💸 Live BTC Price (Binance)", f"${live_price:,.2f}", f"{arrow} {abs(change_percent):.2f}%")

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

        # --- Load full historical BTC data from local CSV ---
        btc_full_df = load_full_btc_data()

        if btc_full_df is not None:
            st.success(f"📚 Loaded full BTC history: {len(btc_full_df)} rows")
            st.dataframe(btc_full_df.tail(3))  # Optional preview


        # Check if 'price' column exists in df
        if 'price' not in df.columns:
            st.error("❌ 'price' column is missing from the data.")
            st.stop()

        # Process the DataFrame
        df = label_trend(df)
        df = create_features(df)

        st.info(f"🧮 Rows after feature engineering: {len(df)}")

        # --- Train or Load Model ---
        # --- Load Pretrained Full History Model ---
        import os
        from btc_trend_classifier import load_custom_model

        model_path = "models/full_history_model.pkl"

        if os.path.exists(model_path):
            model = load_custom_model(model_path)
            st.success("✅ Loaded model trained on full BTC history (2010–2025)")
        else:
            st.error("❌ Full history model not found. Please train it first.")
            st.stop()



        # --- Manual Retrain Button ---
        st.divider()
        st.subheader("🔧 Manual Controls")

        if st.button("🔁 Retrain Model Now"):
            model = train_trend_model(df)
            st.success("✅ Model retrained manually.")






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
    btc_dom, eth_btc, btc_usd, eth_usd = get_altseason_metrics()

    score = 0

    # BTC dominance points
    if btc_dom < 40:
        score += 50
    elif btc_dom < 45:
        score += 30
    elif btc_dom < 50:
        score += 10

    # ETH/BTC ratio points
    if eth_btc > 0.08:
        score += 50
    elif eth_btc > 0.07:
        score += 30
    elif eth_btc > 0.065:
        score += 10

    # Score level
    if score >= 70:
        level = "HIGH"
        msg = "🌊 Altseason likely — altcoins gaining strongly!"
    elif score >= 40:
        level = "MEDIUM"
        msg = "⚠️ Altseason brewing — watch closely."
    else:
        level = "LOW"
        msg = "📉 BTC still dominant — altseason not ready."

    # Display
    st.metric("🧠 Altseason Score", f"{score}%", level)
    st.write(msg)
    st.caption(f"BTC Dominance: {btc_dom:.2f}%")
    st.caption(f"ETH/BTC Ratio: {eth_btc:.4f}")

except Exception as e:
    st.warning(f"⚠️ Could not calculate Altseason Score: {e}")


# --- 7-Day BTC Price Trend ---
st.subheader("📈 7-Day BTC Price Trend")
try:
    trend_df = df[-7:].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df.index, y=trend_df['price'], mode='lines+markers', name='BTC Price'))
    fig.update_layout(template="plotly_dark", title="BTC Price (Last 7 Days)", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True, key="7-day-trend-chart")  # Adding a unique key
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
    st.plotly_chart(fig, use_container_width=True, key="30-day-trend-chart")  # Adding a unique key
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

try:
    # Make sure we have at least 8 valid price points (no NaN)
    df_valid = df[df['price'].notna()]
    
    if len(df_valid) < 8:
        st.warning("⚠️ Need at least 8 valid days of price data to detect bull signals (20%+ gain in 7 days).")
    else:
        df_valid['gain_7d'] = df_valid['price'].pct_change(periods=7)
        bull_df = df_valid[df_valid['gain_7d'] >= 0.2]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['price'], mode='lines', name='BTC Price'))

        if not bull_df.empty:
            fig.add_trace(go.Scatter(x=bull_df.index, y=bull_df['price'], mode='markers',
                                     marker=dict(color='red', size=10), name='20%+ Gain'))

        fig.update_layout(template="plotly_dark",
                          title="BTC Price with Bull Signals (20%+ Gain in 7 Days)",
                          xaxis_title="Date", yaxis_title="Price (USD)")

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

# --- Raw Data Preview ---
with st.expander("📄 View raw data"):
    st.dataframe(df.tail(10))


# --- Save Today's Entry to Google Sheet (Optional Logging to Sheet) ---
def log_today_to_google_sheet():
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials_json = st.secrets["google_credentials"]["value"]
        credentials_dict = json.loads(credentials_json)
        creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open("btc_price_log").worksheet("DailyPrice")

        # Check existing dates to prevent duplicates
        existing_dates = sheet.col_values(1)
        today_str = datetime.now().strftime("%Y-%m-%d")

        if today_str in existing_dates:
            st.info("📌 Today's entry already exists in Google Sheet.")
            return

        # Get the last logged row from df
        last_row = df.iloc[-1]
        new_row = [today_str, round(last_row['price'], 2), round(last_row['change_24h'], 2), round(last_row['change_7d'], 2), last_row['trend']]
        sheet.append_row(new_row)
        st.success("✅ Logged today’s data to Google Sheet.")
    except Exception as e:
        st.warning(f"⚠️ Could not log to Google Sheet: {e}")
