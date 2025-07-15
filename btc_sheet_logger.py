import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import date

# Define the scope and authenticate
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open your sheet by name
sheet = client.open("btc_price_log").sheet1  # Make sure this name matches your actual Google Sheet

# Load existing data from the sheet
data = sheet.get_all_records()
df_sheet = pd.DataFrame(data)

# Prepare today's data
today = date.today().isoformat()
btc_price = 120354.94
change_24h = 1.07
change_7d = 10.55


# Only log if today is not already logged
if df_sheet.empty or "Date" not in df_sheet.columns:
    new_row = [today, btc_price, change_24h, change_7d]
    sheet.append_row(new_row)
    print("✅ BTC data logged (first entry).")
elif today not in df_sheet["Date"].values:
    new_row = [today, btc_price, change_24h, change_7d]
    sheet.append_row(new_row)
    print("✅ BTC data logged to Google Sheets.")
else:
    print("🟡 Already logged today.")
