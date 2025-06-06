# download_dataset.py
import os
import pandas as pd
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET       = "external-lawn-photos"

supabase = create_client(SUPABASE_URL, SERVICE_KEY)

# 1. Load CSV exported earlier
df = pd.read_csv("external_photos.csv")

# 2. For each row, create a folder and download the image via a signed URL
for idx, row in df.iterrows():
    label = row["label"]          # “yard” or “not_yard”
    path  = row["file_path"]      # e.g. "yard_1749222649_1.jpg"
    out_dir = os.path.join("data", label)
    os.makedirs(out_dir, exist_ok=True)

    # 3. Generate a signed URL valid for 1 hour
    signed = supabase.storage.from_(BUCKET).create_signed_url(
        path, 60 * 60  # 3600 seconds = 1 hour
    )["signedURL"]

    # 4. Download the image if not already present
    out_path = os.path.join(out_dir, os.path.basename(path))
    if not os.path.exists(out_path):
        resp = requests.get(signed, timeout=10)
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"↓ Downloaded {label}/{os.path.basename(path)}")

print("✅ All images downloaded into data/yard and data/not_yard")
