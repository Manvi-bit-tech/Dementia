import numpy as np
from scripts.preprocessing import load_dataset
from scripts.train_dl import build_dl_model
from scripts.train_svm import train_svm
from scripts.evaluate import evaluate_model

import joblib
from tensorflow.keras.models import save_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import mode
import os

# Create outputs/models if not exist
os.makedirs("outputs/models", exist_ok=True)

# Load and process data
X, y = load_dataset()
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ✅ Save Label Encoder
joblib.dump(le, "outputs/models/label_encoder.pkl")

# Prepare targets
y_cat = to_categorical(y_encoded)
X = X[..., np.newaxis]  # Add channel dimension for CNN

# Train-test split
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat)
y_train = np.argmax(y_train_cat, axis=1)
y_test = np.argmax(y_test_cat, axis=1)

# DL model
print("[INFO] Training Deep Learning model...")
dl_model = build_dl_model(X_train.shape[1:])
dl_model.fit(X_train, y_train_cat, epochs=25, batch_size=32, verbose=1)

# SVM model
print("[INFO] Training SVM model...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
svm_model = train_svm(X_train_flat, y_train)
svm_preds = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"[RESULT] SVM Accuracy: {svm_accuracy:.4f}")

# Predict using DL model
print("[INFO] Predicting using both models...")
dl_preds_prob = dl_model.predict(X_test)
dl_preds = np.argmax(dl_preds_prob, axis=1)

# Fusion (majority voting)
print("[INFO] Fusing predictions (majority voting)...")
fused_preds = mode([dl_preds, svm_preds], axis=0)[0].flatten()

# Report
target_names = [str(cls) for cls in le.classes_]
print("[RESULT] Fused Accuracy:", accuracy_score(y_test, fused_preds))
print(classification_report(y_test, fused_preds, target_names=target_names))

# Save models ✅
joblib.dump(svm_model, "outputs/models/svm_model.pkl")
dl_model.save("outputs/models/dl_model.keras")  # 🔁 Switch to safer format
print("[INFO] Models saved to outputs/models/")
