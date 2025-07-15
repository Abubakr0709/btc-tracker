import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import streamlit as st  # make sure this is at the top of your file

def load_google_sheet(sheet_name, worksheet_name, raw=False):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name).worksheet(worksheet_name)

    if raw:
        return sheet

    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df.columns = [col.lower().strip() for col in df.columns]

    if 'date' not in df.columns:
        raise KeyError("❌ 'date' column not found in Google Sheet. Check header spelling and casing.")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df




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

def train_trend_model(df):
    X = df[['ret_1d', 'ret_3d', 'ret_7d', 'volatility_7d']]
    y = df['trend']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model



