import os
import json
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO
import open_clip
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client
import time

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

# Initialize CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

def search_images(query, num_results=100):
    params = {
        "engine": "google",
        "q": query,
        "tbm": "isch",
        "num": num_results,
        "api_key": os.getenv('SERPAPI_API_KEY')
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    images = results.get('images_results', [])
    print(f"Found {len(images)} images for query: {query}")
    return images

def download_and_process_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        processed_image = preprocess(image).unsqueeze(0)
        return image, processed_image
    except Exception as e:
        print(f"Error processing image {url}: {str(e)}")
        return None, None

def filter_images(images, threshold=0.2):
    filtered_images = []
    text = torch.cat([
        open_clip.tokenize("a photo of a front yard"),
        open_clip.tokenize("a photo of a backyard"),
        open_clip.tokenize("a photo of a lawn"),
        open_clip.tokenize("a photo of a garden")
    ])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    text = text.to(device)
    
    for img_data in tqdm(images, desc="Filtering images"):
        try:
            image, processed = download_and_process_image(img_data['original'])
            if image is None or processed is None:
                print(f"Skipping image {img_data['original']}: Could not process")
                continue

            with torch.no_grad():
                processed = processed.to(device)
                image_features = model.encode_image(processed)
                text_features = model.encode_text(text)
                similarity = (image_features @ text_features.T).squeeze(0)
                max_similarity = similarity.max().item()
                print(f"Image similarity score: {max_similarity}")

                if max_similarity > threshold:
                    img_data['similarity_score'] = max_similarity
                    filtered_images.append(img_data)

                    try:
                        img_buffer = BytesIO()
                        image.save(img_buffer, format='JPEG')
                        img_buffer.seek(0)

                        filename = f"yard_{int(time.time())}_{len(filtered_images)}.jpg"

                        # ✅ Upload with explicit content-type
                        supabase.storage.from_('external-lawn-photos').upload(
                            path=filename,
                            file=img_buffer.getvalue(),
                            file_options={"content-type": "image/jpeg"}
                        )

                        print(f"✅ Uploaded image to Supabase: {filename}")

                        res = supabase.table('external_photos').insert({
                            'url': img_data['original'],
                            'label': 'yard',
                            'clip_score': max_similarity,
                            'file_path': filename,
                            'source': 'SerpAPI',
                            'search_term': img_data.get('title', 'unknown')
                        }).execute()

                        # ✅ Fix: check for error safely
                        if hasattr(res, 'error') and res.error:
                            print(f"❌ Metadata insert error: {res.error}")
                        else:
                            print("📝 Metadata saved.")

                    except Exception as e:
                        print(f"❌ Error uploading to Supabase: {str(e)}")

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            continue

    return filtered_images

def main():
    queries = [
        "beautiful front yard landscaping",
        "modern front yard design",
        "backyard lawn ideas",
        "perfect lawn care",
        "residential yard landscaping",
        "home garden lawn"
    ]
    
    all_filtered_images = []
    
    for query in queries:
        print(f"\n🔍 Processing query: {query}")
        images = search_images(query)
        filtered = filter_images(images)
        all_filtered_images.extend(filtered)
        time.sleep(2)
    
    print(f"\n✅ Total filtered images: {len(all_filtered_images)}")

if __name__ == "__main__":
    main()
