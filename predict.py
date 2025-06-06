from PIL import Image, ImageStat, ImageFilter
import numpy as np
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def predict(self, image: Path = Input(description="Image of a lawn")) -> float:
        # Resize and convert to RGB
        image = Image.open(str(image)).convert("RGB").resize((224, 224))

        np_img = np.array(image)

        # ---------- GREEN RATIO ----------
        r, g, b = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]
        green_mask = (g > r) & (g > b) & (g > 100)
        green_ratio = np.sum(green_mask) / green_mask.size
        green_score = min(70, green_ratio * 100)

        # ---------- SHARPNESS ----------
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(edges)
        sharpness_score = min(30, stat.stddev[0])

        # ---------- FINAL SCORE ----------
        total_score = round(green_score + sharpness_score)

        print(f"Green %: {green_ratio:.2f} → {green_score:.1f} pts")
        print(f"Sharpness: {stat.stddev[0]:.2f} → {sharpness_score:.1f} pts")
        print(f"Predicted lawn score: {total_score}")

        return total_score

# 👇 Add this block to run directly from CLI
if __name__ == "__main__":
    import sys
    from PIL import Image

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")

    predictor = Predictor()
    predictor.setup()
    predictor.predict(image=image)
