"""
Preprocessing for Diabetic Retinopathy image classification.
Matching the notebook: RGB, resize to IMG_SIZE=128, normalize /255.0
Works with folder structure: data/train/<class>/*.png|jpg
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Must match the notebook (IMG_SIZE = 128)
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IMG_SIZE = 128


def load_images_from_dir(data_dir, img_size=IMG_SIZE, max_per_class=None):
    """Load images from folder structure: data_dir/<class>/*.png|jpg.
    Returns X (N,H,W,3) float32 in [0,1], y (N,) string labels, paths.
    """
    X, y, paths = [], [], []
    if not os.path.isdir(data_dir):
        return np.array([]), np.array([]), []
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        count = 0
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if max_per_class is not None and count >= max_per_class:
                break
            path = os.path.join(class_dir, fname)
            try:
                img = Image.open(path).convert("RGB").resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(label)
                paths.append(path)
                count += 1
            except Exception as e:
                print(f"Skip {path}: {e}")
    return np.array(X), np.array(y), paths


def extract_features(X):
    """Flatten for classical ML models (not used by the best CNN)."""
    if len(X) == 0:
        return np.array([])
    return X.reshape(X.shape[0], -1)


def preprocess_single_image(image_path_or_bytes, img_size=IMG_SIZE):
    """Preprocess one image exactly as done during training / prediction."""
    if isinstance(image_path_or_bytes, (str, os.PathLike)):
        img = Image.open(image_path_or_bytes).convert("RGB")
    else:
        img = Image.open(image_path_or_bytes).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_pipeline(train_dir, test_dir=None, feature_type="image",
                        encoder_path="models/label_encoder.pkl"):
    X_tr, y_tr, _ = load_images_from_dir(train_dir)
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    if feature_type == "flatten":
        X_tr_f = extract_features(X_tr)
    else:
        X_tr_f = X_tr

    if test_dir and os.path.isdir(test_dir) and any(os.listdir(test_dir)):
        X_te, y_te, _ = load_images_from_dir(test_dir)
        y_te_enc = le.transform(y_te)
        X_te_f = extract_features(X_te) if feature_type == "flatten" else X_te
    else:
        X_tr_f, X_te_f, y_tr_enc, y_te_enc = train_test_split(
            X_tr_f, y_tr_enc, test_size=0.2, random_state=42, stratify=y_tr_enc
        )

    os.makedirs(os.path.dirname(encoder_path) or ".", exist_ok=True)
    joblib.dump(le, encoder_path)
    return X_tr_f, X_te_f, y_tr_enc, y_te_enc, le
