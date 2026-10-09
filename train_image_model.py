
import os
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# -----------------------------
# 1. SETTINGS
# -----------------------------
DATA_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/birdy654/"
    "cifake-real-and-ai-generated-synthetic-images/versions/3"
)

MODEL_PATH = "image_model.pth"
BATCH_SIZE = 16
EPOCHS = 3
MAX_TRAIN_IMAGES = 4000
MAX_TEST_IMAGES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. FIND DATASET FOLDERS
# -----------------------------
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        "Could not find train/test folders. Check DATA_DIR."
    )

# -----------------------------
# 3. IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

print("Classes:", train_data.classes)
print("Training images available:", len(train_data))
print("Testing images available:", len(test_data))

if len(train_data.classes) != 2:
    raise ValueError("Expected two classes: AI-generated and real.")

if train_data.classes != test_data.classes:
    raise ValueError("Train and test class names/order do not match.")

# Use a manageable, balanced subset for CPU training.
def balanced_subset(dataset, max_images):
    per_class = max(1, max_images // len(dataset.classes))
    selected = []
    targets = dataset.targets

    for class_index in range(len(dataset.classes)):
        indices = [
            i for i, target in enumerate(targets)
            if target == class_index
        ]
        selected.extend(indices[:per_class])

    return Subset(dataset, selected)

train_subset = balanced_subset(train_data, MAX_TRAIN_IMAGES)
test_subset = balanced_subset(test_data, MAX_TEST_IMAGES)

train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_subset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# -----------------------------
# 4. CREATE MODEL
# -----------------------------
# A small CNN trained from scratch; no extra model download required.
class ImageCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = ImageCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. TRAIN
# -----------------------------
print("\nStarting training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / max(total, 1)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / max(len(train_loader), 1):.4f} | "
        f"Training accuracy: {accuracy:.2f}%"
    )

# -----------------------------
# 6. EVALUATE
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / max(total, 1)
print(f"\nTest accuracy on selected subset: {test_accuracy:.2f}%")

# -----------------------------
# 7. SAVE MODEL
# -----------------------------
torch.save({
    "model_state_dict": model.cpu().state_dict(),
    "classes": train_data.classes
}, MODEL_PATH)

print("Model saved to:", os.path.abspath(MODEL_PATH))