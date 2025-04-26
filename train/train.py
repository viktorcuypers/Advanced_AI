import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from datetime import datetime

def main():
    # --------------------
    # CONFIG
    # --------------------
    DATA_DIR = "datasets/data_filtered" 
    BATCH_SIZE = 192
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    MODEL_PATH = "model/model_multiclass.pth"
    NUM_WORKERS = 6  # For Ryzen 7 5800H

    # --------------------
    # DEVICE SETUP
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Device selected: {'GPU 🧠' if device.type == 'cuda' else 'CPU 🧮'}")

    # --------------------
    # TRANSFORMS
    # --------------------
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


    # --------------------
    # DATASET & DATALOADERS
    # --------------------
    print(f"📂 Loading dataset from: {DATA_DIR}")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes
    print(f"🔍 Found classes: {class_names}")
    print(f"📊 Total images: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"📊 Training images: {len(train_ds)} | Validation images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            pin_memory=True, num_workers=NUM_WORKERS)

    # --------------------
    # MODEL
    # --------------------
    print("🧠 Initializing ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 14)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # --------------------
    # TRAINING LOOP
    # --------------------
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔁 Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  📦 Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"✅ Epoch {epoch+1} complete - Total Loss: {total_loss:.4f} - Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 New best model saved to: {MODEL_PATH}")

    end_time = time.time()
    print(f"\n🎉 Training complete in {(end_time - start_time):.2f} seconds.")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")

# Windows multiprocessing safety
if __name__ == "__main__":
    main()
