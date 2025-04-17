import os
import torch
from torchvision import models, transforms
from PIL import Image

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "model/model_binary.pth"
IMAGE_ROOT = "predict/test_images"
CLASS_NAMES = ["diseased", "healthy"]  # Same as during training

# --------------------
# SETUP
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
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
for label_folder in os.listdir(IMAGE_ROOT):
    label_path = os.path.join(IMAGE_ROOT, label_folder)
    if not os.path.isdir(label_path):
        continue

    print(f"\n🔍 Predicting images in: {label_folder.upper()}")

    for filename in sorted(os.listdir(label_path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(label_path, filename)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(1).item()

        print(f"🖼 {filename}: {CLASS_NAMES[pred].upper()}")
