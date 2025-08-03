import librosa
import numpy as np
import os

def extract_features(file_path, max_len=300):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def load_dataset():
    X, y = [], []
    data_paths = [
        ("data/audio data/control", 0),
        ("data/audio data/dementia", 1)
    ]
    for folder, label in data_paths:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder does not exist: {folder}")
            continue
        for file in os.listdir(folder):
            if file.endswith(".wav") or file.endswith(".mp3"):
                path = os.path.join(folder, file)
                features = extract_features(path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    print(f"[INFO] Loaded {len(X)} samples with {len(y)} labels")
    return np.array(X), np.array(y)