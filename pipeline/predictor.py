# pipeline/predictor.py

import torch
from torchvision import transforms
from PIL import Image
import os

from vision.inference import CXRModel,LABELS
from models import model
# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
MODEL_PATH = "cxr_model.pth"

model = CXRModel(len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    """
    Returns probability tensor for each label
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(image)[0]   # shape: [num_labels]

    return probs
