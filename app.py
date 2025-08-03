import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load the saved model and label encoder
model = load_model("outputs/models/dl_model.h5")
label_encoder = joblib.load("outputs/models/label_encoder.pkl")

st.title("Dementia Detection from Speech")

# Upload audio
audio_file = st.file_uploader("Upload an audio file (.mp3)", type=["mp3"])

if audio_file is not None:
    # Load audio using librosa
    y, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] > 300:
        mfcc = mfcc[:, :300]  # truncate
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])), mode='constant')

    mfcc = mfcc[..., np.newaxis]  # add channel dimension
    mfcc = np.expand_dims(mfcc, axis=0)  # add batch dimension, shape: (1, 40, 300, 1)

    # Predict
    pred = model.predict(mfcc)
    pred_label = np.argmax(pred, axis=1)
    decoded = label_encoder.inverse_transform(pred_label)
    
    st.success(f"Prediction: {decoded[0]}")


