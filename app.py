import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
import streamlit as st

st.set_page_config(
    page_title="Speech & Prosody Analyzer", page_icon="🎙️", layout="wide"
)


@st.cache_resource
def load_speech_model():
  try:
    return joblib.load("speech_svm_model.pkl")
  except FileNotFoundError:
    return None


model = load_speech_model()

st.title("🎙️ Speech Acoustic & Prosodic Emotion Analyzer")
st.markdown(
    "Analyze acoustic speech signals, extract prosodic features (Pitch,"
    " Energy, ZCR), and classify vocal states using **SVM**."
)

if model is None:
  st.warning(
      "Model file `speech_svm_model.pkl` not found. Please run `python"
      " train_model.py` first!"
  )
  st.stop()

# Helper audio feature extractor
def extract_signal_features(signal, sr):
  # Normalize signal
  if signal.ndim > 1:
    signal = signal.mean(axis=1)
  signal = signal / (np.max(np.abs(signal)) + 1e-6)

  # Energy
  energy = np.mean(signal**2)

  # Zero Crossing Rate
  zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2.0

  # Frequency & Pitch proxy via FFT peak
  fft_spectrum = np.abs(np.fft.rfft(signal))
  freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
  peak_freq = freqs[np.argmax(fft_spectrum[1:]) + 1]

  # Spectral Flux proxy
  spec_flux = np.std(fft_spectrum) / (np.mean(fft_spectrum) + 1e-6)

  return np.array([peak_freq, energy, zcr, spec_flux]), signal, sr


# Demo tone generator for instant testing
def generate_sample_audio(tone_type):
  sr = 16000
  duration = 2.0
  t = np.linspace(0, duration, int(sr * duration), endpoint=False)
  if tone_type == "Happy / Excited":
    signal = 0.6 * np.sin(2 * np.pi * 250 * t) + 0.3 * np.random.normal(
        0, 0.05, len(t)
    )
  elif tone_type == "Calm / Low-Energy":
    signal = 0.2 * np.sin(2 * np.pi * 125 * t)
  elif tone_type == "Emphatic":
    signal = 0.8 * np.sin(2 * np.pi * 280 * t) + 0.4 * np.sin(
        2 * np.pi * 560 * t
    )
  else:
    signal = 0.4 * np.sin(2 * np.pi * 165 * t)

  scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
  buf = io.BytesIO()
  wavfile.write(buf, sr, scaled)
  return buf.getvalue()


st.sidebar.header("Audio Input Source")
input_option = st.sidebar.radio(
    "Select Input Method:", ["Upload .WAV Audio", "Use Built-in Audio Samples"]
)

audio_data = None

if input_option == "Upload .WAV Audio":
  uploaded_file = st.sidebar.file_uploader(
      "Choose a .wav file", type=["wav", "wave"]
  )
  if uploaded_file is not None:
    audio_data = uploaded_file.read()
else:
  sample_choice = st.sidebar.selectbox(
      "Choose a Sample Tone:",
      ["Neutral", "Happy / Excited", "Calm / Low-Energy", "Emphatic"],
  )
  if st.sidebar.button("Load Selected Sample"):
    audio_data = generate_sample_audio(sample_choice)

if audio_data is not None:
  st.audio(audio_data, format="audio/wav")

  # Process signal
  sr, raw_signal = wavfile.read(io.BytesIO(audio_data))
  feats, clean_sig, sr = extract_signal_features(raw_signal, sr)

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Fundamental Freq (Pitch)", f"{feats[0]:.1f} Hz")
  col2.metric("RMS Energy", f"{feats[1]:.4f}")
  col3.metric("Zero-Crossing Rate", f"{feats[2]:.3f}")
  col4.metric("Spectral Dispersion", f"{feats[3]:.2f}")

  st.write("---")

  # Visualizations
  st.subheader("Acoustic Signal Waveform & Spectrogram")
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
  time_axis = np.linspace(0, len(clean_sig) / sr, num=len(clean_sig))

  ax1.plot(time_axis, clean_sig, color="#1f77b4", lw=0.8)
  ax1.set_ylabel("Amplitude")
  ax1.set_title("Time-Domain Waveform")
  ax1.grid(True, alpha=0.3)

  ax2.specgram(clean_sig, Fs=sr, NFFT=512, noverlap=256, cmap="viridis")
  ax2.set_xlabel("Time (seconds)")
  ax2.set_ylabel("Frequency (Hz)")
  ax2.set_title("Spectrogram (Energy Distribution)")
  plt.tight_layout()
  st.pyplot(fig)

  # Classification
  class_labels = [
      "Neutral",
      "Happy / Excited",
      "Calm / Low-Energy",
      "Emphatic",
  ]
  features_vector = feats.reshape(1, -1)
  prediction = model.predict(features_vector)[0]
  probs = model.predict_proba(features_vector)[0]

  st.write("---")
  st.subheader("Prosody Prediction & Confidence")
  st.success(f"🎯 **Predicted State: {class_labels[prediction]}**")

  prob_df = pd.DataFrame({"Prosodic Class": class_labels, "Probability": probs})
  st.bar_chart(prob_df.set_index("Prosodic Class"))
else:
  st.info("👈 Select a sample from the sidebar or upload a `.wav` recording.")