import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def generate_acoustic_dataset(n_samples=1800):
  np.random.seed(42)
  # Emotion classes: 0: Neutral, 1: Happy / Excited, 2: Somber / Low-Energy, 3: Emphatic / High-Stress
  labels = np.random.choice([0, 1, 2, 3], size=n_samples)

  features = []
  for label in labels:
    if label == 0:  # Neutral
      pitch = np.random.normal(160, 20)
      energy = np.random.normal(0.04, 0.01)
      zcr = np.random.normal(0.08, 0.02)
      spec_flux = np.random.normal(1.2, 0.3)
    elif label == 1:  # Happy / Excited
      pitch = np.random.normal(240, 30)
      energy = np.random.normal(0.09, 0.02)
      zcr = np.random.normal(0.15, 0.03)
      spec_flux = np.random.normal(2.5, 0.5)
    elif label == 2:  # Somber / Calm
      pitch = np.random.normal(120, 15)
      energy = np.random.normal(0.02, 0.005)
      zcr = np.random.normal(0.04, 0.01)
      spec_flux = np.random.normal(0.8, 0.2)
    else:  # Emphatic / Stressed (ADsPro prominence related)
      pitch = np.random.normal(270, 35)
      energy = np.random.normal(0.12, 0.03)
      zcr = np.random.normal(0.18, 0.04)
      spec_flux = np.random.normal(3.1, 0.6)

    features.append([pitch, energy, zcr, spec_flux])

  return np.array(features), labels


def train_speech_svm():
  X, y = generate_acoustic_dataset()
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Train Support Vector Classifier with RBF Kernel
  pipeline = Pipeline([
      ("scaler", StandardScaler()),
      ("svm", SVC(kernel="rbf", C=2.0, probability=True, random_state=42)),
  ])

  pipeline.fit(X_train, y_train)

  y_pred = pipeline.predict(X_test)
  target_names = ["Neutral", "Happy / Excited", "Calm / Low-Energy", "Emphatic"]
  print("Speech Classifier Performance Report:\n")
  print(
      classification_report(y_test, y_pred, target_names=target_names, digits=3)
  )

  joblib.dump(pipeline, "speech_svm_model.pkl")
  print("Model saved as 'speech_svm_model.pkl'.")


if __name__ == "__main__":
  train_speech_svm()