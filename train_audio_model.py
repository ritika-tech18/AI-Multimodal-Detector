
import os
import io
from collections import Counter

import torch
import torch.nn as nn
import librosa
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio

# Settings
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "audio_model.pth"
)

BATCH_SIZE = 8
EPOCHS = 3
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)


# Audio feature extraction
def extract_features(audio):
    audio = np.asarray(audio, dtype=np.float32)

    target_length = SAMPLE_RATE * DURATION

    if len(audio) == 0:
        audio = np.zeros(target_length, dtype=np.float32)

    audio = audio[:target_length]

    if len(audio) < target_length:
        audio = np.pad(
            audio, (0, target_length - len(audio))
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80.0) / 80.0

    return torch.tensor(
        mel, dtype=torch.float32
    ).unsqueeze(0)


# Dataset: read raw bytes without TorchCodec
class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.cast_column(
            "audio", Audio(decode=False)
        )

        self.labels = [
            int(row["label"]) for row in self.dataset
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        item = row["audio"]

        raw_bytes = item.get("bytes")

        if raw_bytes is None:
            raise ValueError(
                f"Audio bytes missing at index {index}"
            )

        waveform, sample_rate = librosa.load(
            io.BytesIO(raw_bytes),
            sr=SAMPLE_RATE,
            mono=True
        )

        features = extract_features(waveform)

        return features, torch.tensor(
            self.labels[index], dtype=torch.long
        )


# CNN model
class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# Load dataset
print("Loading audio dataset...")

raw_dataset = load_dataset("Hemg/Deepfake-Audio-Dataset")

print("Available splits:", list(raw_dataset.keys()))

train_data = AudioDataset(raw_dataset["train"])

print("Training examples:", len(train_data))
print("Label distribution:", Counter(train_data.labels))

if len(set(train_data.labels)) < 2:
    raise ValueError(
        "Training data contains fewer than two classes."
    )

print("Testing one audio sample...")

sample_features, sample_label = train_data[0]

print("Feature shape:", tuple(sample_features.shape))
print("Sample label:", sample_label.item())

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Train model
model = AudioCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / max(total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / max(len(train_loader), 1):.4f} | "
        f"Training accuracy: {accuracy:.2f}%"
    )

# Save without assuming the real/fake label mapping
torch.save(
    {
        "model_state_dict": model.cpu().state_dict(),
        "classes": ["LABEL_0", "LABEL_1"],
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_mels": N_MELS
    },
    MODEL_PATH
)

print("Audio model saved to:", MODEL_PATH)