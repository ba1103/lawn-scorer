import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# Load env vars
load_dotenv()

# Init Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# Query all records
res = supabase.table("external_photos").select("*").execute()

# Save to CSV
if hasattr(res, 'data') and res.data:
    df = pd.DataFrame(res.data)
    df.to_csv("external_photos.csv", index=False)
    print("✅ Exported to external_photos.csv")
else:
    print("⚠️ No data found or error in response.")
