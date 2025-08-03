import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import soundfile as sf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

# ========== LOAD MODEL & ENCODER ==========
model = load_model("outputs/models/dl_model.keras", compile=False)
label_encoder = joblib.load("outputs/models/label_encoder.pkl")

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Dementia Detection App", layout="centered")
st.title("🧠 Dementia Detection from Speech")
st.image(
    "https://www.researchgate.net/publication/332790561/figure/fig1/AS:766771086766082@1559823881581/Cookie-Theft-Picture-4.ppm",
    caption="🖼️ Describe this Cookie Theft picture in your own words."
)
st.markdown("---")

# ========== AUDIO PREPROCESSING FUNCTION ==========
def preprocess_audio(path):
    y, sr = librosa.load(path, sr=16000)

    # Trim silence and normalize
    y, _ = librosa.effects.trim(y)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] > 300:
        mfcc = mfcc[:, :300]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])), mode='constant')

    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Shape: (1, 40, 300, 1)
    return mfcc

# ========== RECORD AUDIO ==========
st.header("🎤 Record Your Audio")

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio = frame.to_ndarray()
        self.samplerate = frame.sample_rate
        return frame

ctx = webrtc_streamer(
    key="record",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx and ctx.audio_processor:
    if st.button("🧪 Predict from Recorded Audio"):
        st.info("⏳ Processing recorded audio...")
        try:
            audio = ctx.audio_processor.audio
            sr = ctx.audio_processor.samplerate

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio.T, sr)
                tmp_path = tmpfile.name

            mfcc = preprocess_audio(tmp_path)
            st.write("✅ MFCC shape (recorded):", mfcc.shape)

            pred = model.predict(mfcc)
            confidence = np.max(pred)
            pred_label = np.argmax(pred, axis=1)
            decoded = label_encoder.inverse_transform(pred_label)

            st.success(f"🎙️ Prediction from recorded audio: **{decoded[0]}** (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(f"⚠️ Error processing audio: {e}")

st.markdown("---")

# ========== UPLOAD AUDIO ==========
st.subheader("📁 Or Upload an Existing Audio File (.mp3)")

audio_file = st.file_uploader("Upload an audio file", type=["mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        mfcc = preprocess_audio(tmp_path)
        st.write("✅ MFCC shape (uploaded):", mfcc.shape)

        pred = model.predict(mfcc)
        confidence = np.max(pred)
        pred_label = np.argmax(pred, axis=1)
        decoded = label_encoder.inverse_transform(pred_label)

        st.success(f"📤 Prediction from uploaded audio: **{decoded[0]}** (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"⚠️ Error processing uploaded file: {e}")

