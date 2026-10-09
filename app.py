
import os
import io
import tempfile

import streamlit as st
import joblib
import librosa
import numpy as np
import cv2
import torch

from torch import nn
from torchvision import transforms
from PIL import Image


# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="AI DetectX",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================
# LOAD TEXT MODEL
# =====================================

@st.cache_resource
def load_text_model():
    return joblib.load(
        os.path.join(BASE_DIR, "text_model.joblib")
    )


# =====================================
# LOAD AUDIO MODEL
# =====================================

@st.cache_resource
def load_audio_model():
    return joblib.load(
        os.path.join(BASE_DIR, "audio_model.joblib")
    )


def predict_audio(uploaded_audio):
    waveform, sr = librosa.load(
        io.BytesIO(uploaded_audio.getvalue()),
        sr=16000,
        mono=True
    )

    if len(waveform) == 0:
        raise ValueError("The audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=40
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ]).reshape(1, -1)

    model = load_audio_model()
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    fake_score = float(probabilities[classes.index(0)])
    real_score = float(probabilities[classes.index(1)])

    return prediction, fake_score, real_score


# =====================================
# IMAGE MODEL ARCHITECTURE
# Must match the trained CNN
# =====================================

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


# =====================================
# LOAD IMAGE MODEL
# =====================================

@st.cache_resource
def load_image_model():
    path = os.path.join(BASE_DIR, "image_model.pth")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = ImageCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["classes"]


image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =====================================
# COMMON IMAGE PREDICTION
# =====================================

def predict_image(image):
    model, classes = load_image_model()
    tensor = image_transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    scores = {
        str(classes[i]).upper(): float(probabilities[i])
        for i in range(len(classes))
    }

    fake_score = scores.get("FAKE", 0.0)
    real_score = scores.get("REAL", 0.0)

    return fake_score, real_score


# =====================================
# VIDEO PREDICTION
# =====================================

def predict_video(uploaded_video):
    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not open the video. Try an MP4 file."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("No readable video frames were found.")

        indices = np.linspace(
            0,
            total_frames - 1,
            min(10, total_frames),
            dtype=int
        )

        ai_scores = []
        real_scores = []
        sampled_frames = []

        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            fake_score, real_score = predict_image(pil_image)

            ai_scores.append(fake_score)
            real_scores.append(real_score)
            sampled_frames.append(pil_image)

        if not ai_scores:
            raise ValueError("Could not extract any usable frames.")

        return (
            float(np.mean(ai_scores)),
            float(np.mean(real_scores)),
            sampled_frames
        )

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================
# WEBSITE HEADER
# =====================================

st.title("🤖 AI DetectX")
st.subheader("Multimodal AI-Generated Content Detection System")

st.write(
    "Analyze text, images, audio, and video frames using "
    "machine-learning models."
)

st.info(
    "AI DetectX is a research prototype. Results may be incorrect "
    "and are not proof of content authenticity or authorship."
)

st.divider()

content_type = st.radio(
    "Choose content to analyze:",
    ["Text", "Image", "Audio", "Video"],
    horizontal=True
)


# =====================================
# TEXT DETECTION
# =====================================

if content_type == "Text":
    st.header("📝 Text Detection")

    text = st.text_area(
        "Enter or paste your text:",
        height=200
    )

    if st.button("Analyze Text"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                model = load_text_model()
                scores = model.predict_proba([text])[0]
                classes = list(model.classes_)

                human_score = (
                    float(scores[classes.index(0)])
                    if 0 in classes else 0.0
                )
                ai_score = (
                    float(scores[classes.index(1)])
                    if 1 in classes else 0.0
                )

                if max(human_score, ai_score) < 0.60:
                    st.warning("Uncertain result")
                elif ai_score > human_score:
                    st.error("Prediction: AI-generated text")
                else:
                    st.success("Prediction: Human-written text")

                col1, col2 = st.columns(2)
                col1.metric(
                    "Human model score",
                    f"{human_score * 100:.1f}%"
                )
                col2.metric(
                    "AI model score",
                    f"{ai_score * 100:.1f}%"
                )

            except Exception as e:
                st.error(f"Text model error: {e}")


# =====================================
# IMAGE DETECTION
# =====================================

elif content_type == "Image":
    st.header("🖼️ Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", width=400)

            if st.button("Analyze Image"):
                fake_score, real_score = predict_image(image)

                if max(fake_score, real_score) < 0.60:
                    st.warning("Uncertain result")
                elif fake_score > real_score:
                    st.error("Prediction: AI-generated image")
                else:
                    st.success("Prediction: Real image")

                col1, col2 = st.columns(2)
                col1.metric(
                    "AI-generated score",
                    f"{fake_score * 100:.1f}%"
                )
                col2.metric(
                    "Real-image score",
                    f"{real_score * 100:.1f}%"
                )

        except Exception as e:
            st.error(f"Image model error: {e}")


# =====================================
# AUDIO DETECTION
# =====================================

elif content_type == "Audio":
    st.header("🎵 Audio Detection")

    st.write("Upload an audio file for experimental analysis.")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        key="audio_uploader"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("Detect Audio"):
            try:
                with st.spinner("Analyzing audio..."):
                    prediction, fake_score, real_score = predict_audio(
                        uploaded_audio
                    )

                col1, col2 = st.columns(2)
                col1.metric(
                    "AI-generated / Fake score",
                    f"{fake_score * 100:.2f}%"
                )
                col2.metric(
                    "Real / Human score",
                    f"{real_score * 100:.2f}%"
                )

                if max(fake_score, real_score) < 0.60:
                    st.warning("Uncertain result")
                elif prediction == 0:
                    st.error("Prediction: AI-generated / Synthetic Audio")
                else:
                    st.success("Prediction: Real / Human Audio")

                st.caption(
                    "The audio model was trained on a small dataset. "
                    "Its output may not generalize to unfamiliar recordings."
                )

            except Exception as e:
                st.error(f"Audio analysis failed: {e}")


# =====================================
# VIDEO DETECTION
# =====================================

elif content_type == "Video":
    st.header("🎥 Video Detection")

    st.write(
        "Upload a video to analyze up to 10 sampled frames "
        "using the existing image model."
    )

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Analyze Video"):
            try:
                with st.spinner("Extracting and analyzing frames..."):
                    ai_score, real_score, frames = predict_video(
                        uploaded_video
                    )

                st.subheader("Video Frame Analysis")

                col1, col2 = st.columns(2)
                col1.metric(
                    "Average AI-generated frame score",
                    f"{ai_score * 100:.2f}%"
                )
                col2.metric(
                    "Average real-image frame score",
                    f"{real_score * 100:.2f}%"
                )

                if max(ai_score, real_score) < 0.60:
                    st.warning("Uncertain result")
                elif ai_score > real_score:
                    st.error(
                        "Frames classified more strongly as AI-generated"
                    )
                else:
                    st.success(
                        "Frames classified more strongly as real images"
                    )

                st.subheader("Sampled Frames")

                for start in range(0, len(frames), 3):
                    columns = st.columns(3)

                    for column, frame in zip(
                        columns,
                        frames[start:start + 3]
                    ):
                        column.image(
                            frame,
                            use_container_width=True
                        )

                st.caption(
                    "This prototype analyzes selected frames only. "
                    "It does not detect audio deepfakes or verify the "
                    "authenticity of the entire video."
                )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")


# =====================================
# FOOTER
# =====================================

st.divider()
st.caption("AI DetectX | Multimodal AI Content Detection System")