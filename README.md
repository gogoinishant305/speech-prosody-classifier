# Speech Acoustic & Prosodic Emotion Analyzer

An end-to-end Machine Learning audio analysis web application built with **Python**, **Scikit-Learn (SVM)**, **SciPy**, and **Streamlit**.

## Features
- **Acoustic Feature Extraction:** Computes pitch proxy, zero-crossing rate (ZCR), signal energy, and spectral dispersion.
- **Support Vector Machine (SVM):** Classifies speech into acoustic emotional states (Neutral, Excited, Calm, Emphatic).
- **Signal Visualizations:** Real-time waveform rendering and frequency spectrogram display.
- **Dual Mode Input:** Upload `.wav` audio files or test against pre-synthesized acoustic audio samples.

## Quickstart

1. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. Train the SVM classification pipeline:
   \`\`\`bash
   python train_model.py
   \`\`\`

3. Launch the dashboard:
   \`\`\`bash
   python -m streamlit run app.py
   \`\`\`