# vision/inference.py

import torch
from torchvision import transforms
from PIL import Image

from models.model import CXRModel   # ✔ works because model.py is at root

LABELS = [
    "cardiomegaly",
    "pneumonia",
    "pleural_effusion",
    "edema",
    "atelectasis",
    "consolidation",
    "no_finding"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image_path):
    model = CXRModel(len(LABELS))
    model.load_state_dict(torch.load("cxr_model.pth", map_location="cpu"))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = model(image).squeeze().tolist()

    return dict(zip(LABELS, probs))
