# train_classifier.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# 1. Define paths and hyperparameters
DATA_DIR    = "data"       # must contain subfolders 'yard' and 'not_yard'
BATCH_SIZE  = 32
NUM_EPOCHS  = 5
NUM_CLASSES = 2            # yard vs not_yard

# 2. Image transforms for training/validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Create a single dataset using ImageFolder
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
#    Expects:
#      data/yard/*
#      data/not_yard/*

# 4. Split into 80% train / 20% validation
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 5. Load pretrained ResNet-18 and replace final layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. Training & validation loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / train_size
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"           Val accuracy: {acc:.4f}")

# 7. Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/yard_classifier_resnet18.pth")
print("✅ Model saved to models/yard_classifier_resnet18.pth")
