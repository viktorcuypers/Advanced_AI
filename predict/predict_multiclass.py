import os
import torch
from torchvision import models, transforms
from PIL import Image

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "model/model_multiclass.pth"
IMAGE_ROOT = "predict/test_images"
CLASS_NAMES = [
    'bacterial_spot', 'black_rot', 'cedar_apple_rust', 'early_blight',
    'healthy', 'late_blight', 'leaf_mold', 'leaf_scorch',
    'northern_leaf_blight', 'powdery_mildew', 'septoria_leaf_spot',
    'target_spot', 'tomato_mosaic_virus', 'tomato_yellow_leaf_curl_virus'
]

# --------------------
# SETUP
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------
# PREDICT
# --------------------
print(f"📂 Scanning folder: {IMAGE_ROOT}")

# Walk through all subfolders
for root, _, files in os.walk(IMAGE_ROOT):
    for filename in sorted(files):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(root, filename)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(1).item()

        print(f"🖼 {filename}: {CLASS_NAMES[pred].upper()}")
