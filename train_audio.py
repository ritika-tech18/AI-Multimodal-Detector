
from datasets import load_dataset, Audio
import io
import librosa
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading dataset...")

dataset = load_dataset(
    "Hemg/Deepfake-Audio-Dataset",
    split="train"
)

dataset = dataset.cast_column("audio", Audio(decode=False))

features = []
labels = []

print("Extracting MFCC features from audio files...")

for i, row in enumerate(dataset):
    try:
        audio_bytes = row["audio"]["bytes"]

        waveform, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        # Extract 40 MFCC coefficients
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=40
        )

        # Mean and standard deviation of each coefficient
        mean_features = np.mean(mfcc, axis=1)
        std_features = np.std(mfcc, axis=1)

        combined_features = np.concatenate(
            [mean_features, std_features]
        )

        features.append(combined_features)
        labels.append(int(row["label"]))

        print(f"Processed {i + 1}/{len(dataset)}")

    except Exception as e:
        print(f"Skipping file {i}: {e}")

X = np.array(features)
y = np.array(labels)

print("\nTotal usable audio files:", len(y))
print("Feature shape:", X.shape)

if len(np.unique(y)) != 2:
    raise ValueError("Dataset must contain both audio classes.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train the classifier
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)

print("\nTraining audio detection model...")
model.fit(X_train, y_train)

# Evaluate the model on held-out files
predictions = model.predict(X_test)

print("\n========== AUDIO MODEL RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake Audio", "Real Audio"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    os.path.join("models", "audio_model.joblib")
)

print("\nModel saved to models/audio_model.joblib")
print("Audio model training completed!")