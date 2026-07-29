"""Utilities for loading the saved DR model and making a single-image prediction.
Uses the same preprocessing as the notebook (128x128 RGB /255).
"""
import os
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from .preprocessing import IMG_SIZE, preprocess_single_image
from .keras_compat import load_keras_model_compat


def load_artifacts(model_path="models/best_model.keras", encoder_path="models/label_encoder.pkl"):
    """Load the best Keras model (Advanced CNN / MobileNetV2) and label encoder."""
    if not os.path.exists(model_path):
        # fallback names
        alt = model_path.replace("best_model.keras", "best_model_cnn.keras")
        if os.path.exists(alt):
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    # Compatibility loader handles Keras-3 → Keras-2 config differences
    model = load_keras_model_compat(model_path, compile=False)
    le = joblib.load(encoder_path)
    return model, le


def predict_single_image(image_path_or_bytes, model, le, img_size=IMG_SIZE):
    """Predict severity for one fundus image. Returns (label, confidence, prob_dict)."""
    feats = preprocess_single_image(image_path_or_bytes, img_size=img_size)
    probs = model.predict(feats, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = le.inverse_transform([pred_idx])[0]
    conf = float(np.max(probs))
    # Build probability dict using encoder classes order
    classes = list(le.classes_)
    prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
    return label, conf, prob_dict