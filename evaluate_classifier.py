import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# 1) Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # binary: 0=non-yard, 1=yard
checkpoint = torch.load("models/yard_classifier_resnet18.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 2) Transforms should match what you used in training
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3) Load test dataset
test_dataset = datasets.ImageFolder("data/test", transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4) Run inference over test set
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5) Compute and print metrics
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

print("Classes:", test_dataset.classes)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
