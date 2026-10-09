import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD HC3 DATASET
# ==========================================

url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"

print("Loading HC3 dataset...")

df = pd.read_json(url, lines=True)

print("Dataset loaded successfully!")
print("Number of questions:", len(df))


# ==========================================
# 2. CREATE HUMAN + AI TEXT DATA
# ==========================================

texts = []
labels = []

for _, row in df.iterrows():

    # Human answers
    human_answers = row["human_answers"]

    if isinstance(human_answers, list):
        for answer in human_answers:
            if isinstance(answer, str) and answer.strip():
                texts.append(answer)
                labels.append(0)   # 0 = Human

    # ChatGPT answers
    ai_answers = row["chatgpt_answers"]

    if isinstance(ai_answers, list):
        for answer in ai_answers:
            if isinstance(answer, str) and answer.strip():
                texts.append(answer)
                labels.append(1)   # 1 = AI


data = pd.DataFrame({
    "text": texts,
    "label": labels
})


print("\nTraining data created!")
print("Total text samples:", len(data))

print("\nClass distribution:")
print(data["label"].value_counts())


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE MODEL
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining AI text detection model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 6. TEST MODEL
# ==========================================

print("\nTesting model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Human", "AI"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ==========================================
# 7. SAVE MODEL
# ==========================================

model_path = "text_model.joblib"

joblib.dump(model, model_path)

print("\n====================================")
print("MODEL SAVED SUCCESSFULLY!")
print("File:", model_path)
print("====================================")