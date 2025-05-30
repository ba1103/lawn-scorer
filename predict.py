from cog import BasePredictor, Input
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        self.model.eval()

    def predict(self, image: Image = Input(description="Image of a lawn")) -> float:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor).item()
        score = max(0, min(100, round(output)))
        return score
