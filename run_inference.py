import os
import torch
from torchvision import transforms, models
from supabase import create_client
from urllib.parse import urljoin
import requests
from io import BytesIO
from PIL import Image

# ── Configuration ─────────────────────────────────
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME   = "yard-images"   # ← use your public bucket
IMAGE_BASEURL = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/"
MODEL_PATH    = "models/yard_classifier_resnet18.pth"

# ── Initialize Supabase client ───────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Load the trained model ───────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 0 = not_yard, 1 = yard
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def download_image_from_supabase(path: str) -> torch.Tensor:
    """
    Download a PUBLIC image from yard-images bucket and return a preprocessed tensor.
    `path` is e.g. "uploads/abc123.jpg"
    """
    url = IMAGE_BASEURL + path
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return preprocess(img).unsqueeze(0)  # shape [1,3,224,224]

def run_inference():
    # Fetch rows in yard_photos where is_valid_yard IS NULL
    res = supabase.table("yard_photos")\
        .select("id, file_path")\
        .eq("is_valid_yard", None)\
        .execute()

    records = res.data
    if not records:
        print("No new images to process.")
        return

    for rec in records:
        photo_id  = rec["id"]
        file_path = rec["file_path"]  # e.g. "uploads/abc123.jpg"

        try:
            input_tensor = download_image_from_supabase(file_path).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
                confidence, pred_class = torch.max(probs, dim=0)
                is_yard = bool(pred_class.item())  # 1 => yard, 0 => not_yard

            # Update yard_photos row
            update_res = supabase.table("yard_photos")\
                .update({
                    "is_valid_yard": is_yard,
                    "classifier_confidence": float(confidence.item())
                })\
                .eq("id", photo_id)\
                .execute()

            if update_res.error:
                print(f"❌ Failed to update yard_photos for id={photo_id}: {update_res.error}")
            else:
                print(f"✅ Photo {photo_id} labeled as {'yard' if is_yard else 'not_yard'} ({confidence:.2f})")

            # Insert into classifier_results
            insert_res = supabase.table("classifier_results").insert({
                "yard_photo_id": photo_id,
                "label": "yard" if is_yard else "not_yard",
                "confidence": float(confidence.item())
            }).execute()
            if insert_res.error:
                print(f"❌ Failed to insert into classifier_results: {insert_res.error}")

        except Exception as e:
            print(f"❌ Error processing photo_id={photo_id}: {e}")

if __name__ == "__main__":
    run_inference()

