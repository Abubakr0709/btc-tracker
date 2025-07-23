from google.oauth2.service_account import Credentials  # ✅ correct import
import pandas as pd
import gspread
from google.auth.transport.requests import Request
import streamlit as st

def load_google_sheet(sheet_name, worksheet_name, credentials_dict):

    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        client = gspread.authorize(creds)
        sheet = client.open(sheet_name).worksheet(worksheet_name)

        data = sheet.get_all_records()
        if not data:
            st.error("❌ Google Sheet is empty or headers are missing.")
            return None

        df = pd.DataFrame(data)
        df.columns = [col.lower().strip() for col in df.columns]

        if 'date' not in df.columns:
            st.error("❌ 'date' column missing in Google Sheet.")
            return None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.set_index('date', inplace=True)

        for col in ['price', 'change_24h', 'change_7d']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        import traceback
        st.error(f"❌ Failed to load Google Sheet:\n{traceback.format_exc()}")
        return None


def label_trend(df, window=7, up_thresh=0.04, down_thresh=-0.04):
    df['7d_return'] = df['price'].pct_change(periods=window)
    
    def trend_label(x):
        if x >= up_thresh:
            return "Uptrend"
        elif x <= down_thresh:
            return "Downtrend"
        else:
            return "Sideways"
    
    df['trend'] = df['7d_return'].apply(trend_label)
    return df


def create_features(df):
    df['ret_1d'] = df['price'].pct_change(1)
    df['ret_3d'] = df['price'].pct_change(3)
    df['ret_7d'] = df['price'].pct_change(7)
    df['volatility_7d'] = df['ret_1d'].rolling(7).std()
    df.dropna(inplace=True)
    return df


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

MODEL_PATH = "models/random_forest.pkl"

def train_trend_model(df, save=True, model_path="models/model.pkl"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    import os

    X = df[['ret_1d', 'ret_3d', 'ret_7d', 'volatility_7d']]
    y = df['trend']


    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    if save:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

    return model



def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

import joblib

def load_custom_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {path}: {e}")
        return None




