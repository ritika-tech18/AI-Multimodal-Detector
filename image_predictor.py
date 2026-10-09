
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms


# Use the same architecture as train_image_model.py
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "image_model.pth"
)

DEVICE = torch.device("cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def load_image_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]

    if len(classes) != 2:
        raise ValueError("Expected exactly two image classes.")

    model = ImageCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes


def predict_image(image, model, classes):
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(probabilities.argmax().item())

    return {
        "class": classes[index],
        "confidence": float(probabilities[index].item()) * 100,
        "scores": {
            classes[i]: float(probabilities[i].item()) * 100
            for i in range(len(classes))
        },
    }
