from dotenv import load_dotenv
import os
import json

load_dotenv()

creds_raw = os.getenv("GOOGLE_CREDENTIALS_JSON")
if not creds_raw:
    print("❌ GOOGLE_CREDENTIALS_JSON not found")
else:
    try:
        creds = json.loads(creds_raw)
        print("✅ GOOGLE_CREDENTIALS_JSON loaded successfully")
    except json.JSONDecodeError as e:
        print("❌ JSON Decode Error:", e)
