import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import transforms, models
from supabase import create_client
from io import BytesIO
import requests
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()  # so we can read SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from .env

# ── Configuration ─────────────────────────────────────────────────────────
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME   = "external-lawn-photos"   # ← your public bucket name
MODEL_PATH    = "models/yard_classifier_resnet18.pth"

IMAGE_BASEURL = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/"
# Public bucket means we can GET: IMAGE_BASEURL + file_path

# ── Initialize Supabase client ─────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── FastAPI setup ─────────────────────────────────────────────────────────
app = FastAPI()

# Enable CORS so your Expo app (running on a different origin) can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # you can restrict this to your Expo dev URL if desired
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    file_path: str   # e.g. "uploads/abc123.jpg"

class InferenceResponse(BaseModel):
    is_valid_yard: bool
    confidence: float

# ── Load the trained model once at startup ─────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # classes: [0]=not_yard, [1]=yard

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Transforms must match exactly what you used during training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Helper to download a PUBLIC image from Supabase storage ─────────────────
def download_image_from_supabase(file_path: str) -> Image.Image:
    """
    Downloads from your public bucket at:
      https://<SUPABASE_URL>/storage/v1/object/public/{BUCKET_NAME}/{file_path}
    Returns a PIL.Image in RGB mode.
    """
    url = IMAGE_BASEURL + file_path
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Image not found")
    return Image.open(BytesIO(resp.content)).convert("RGB")

# ── /infer endpoint ──────────────────────────────────────────────────────────
@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    # 1) Download the image from the public bucket
    try:
        img = download_image_from_supabase(req.file_path)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2) Preprocess & prepare a batch of size 1
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]

    # 3) Run the model
    with torch.no_grad():
        outputs = model(input_tensor)  # shape [1,2]
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0)
        confidence, pred_class = torch.max(probs, dim=0)
        is_yard = bool(pred_class.item())  # True if class index == 1

    # 4) Return the result
    return InferenceResponse(is_valid_yard=is_yard, confidence=float(confidence.item()))
