"""
FastAPI – Diabetic Retinopathy Severity Classifier
Endpoints: /health, /predict, /upload_retrain, /retrain, /metrics, /dataset_stats, /retrain_log
Best model = Advanced CNN (MobileNetV2 transfer learning) saved as models/best_model.keras
Preprocessing MUST match notebook: RGB, 128x128, /255.0
"""
import os
import time
import uuid
import sys
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
import io
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import IMG_SIZE, CLASSES, load_images_from_dir, preprocess_single_image
from src.keras_compat import load_keras_model_compat

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.keras")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.pkl")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
RETRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "retrain_buffer")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RETRAIN_DIR, exist_ok=True)

START_TIME = time.time()
RETRAIN_LOG = []

app = FastAPI(
    title="Diabetic Retinopathy MLOps API",
    version="1.0.0",
    description="Severity classification (No_DR → Proliferate_DR) from fundus images",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_le = None


def get_model():
    global _model, _le
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, "Model not found. Place best_model.keras in models/.")
        # Compatibility loader handles Keras-3 → Keras-2 config differences
        _model = load_keras_model_compat(MODEL_PATH, compile=False)
        _le = joblib.load(ENCODER_PATH)
    return _model, _le


def reload_model():
    global _model, _le
    _model = load_keras_model_compat(MODEL_PATH, compile=False)
    _le = joblib.load(ENCODER_PATH)


@app.get("/")
def root():
    return {"message": "Diabetic Retinopathy Severity API", "docs": "/docs"}


@app.get("/health")
def health():
    model_ok = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy" if model_ok else "degraded",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_loaded": model_ok,
        "img_size": IMG_SIZE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single-image prediction. Preprocessing identical to notebook training."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    try:
        feats = preprocess_single_image(io.BytesIO(contents), img_size=IMG_SIZE)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    model, le = get_model()
    probs = model.predict(feats, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = str(le.inverse_transform([pred_idx])[0])
    conf = float(np.max(probs))
    classes = list(le.classes_)
    prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}

    # Save upload for audit
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    with open(os.path.join(UPLOAD_DIR, fname), "wb") as f:
        f.write(contents)

    return {
        "prediction": label,
        "confidence": conf,
        "probabilities": prob_dict,
        "filename": file.filename,
    }


@app.post("/upload_retrain")
async def upload_retrain(files: List[UploadFile] = File(...), labels: str = Form(...)):
    """
    Upload one or more fundus images + matching labels (comma-separated).
    Images are saved under data/retrain_buffer/<label>/ for later retraining.
    """
    label_list = [l.strip() for l in labels.split(",")]
    if len(label_list) != len(files):
        raise HTTPException(400, "Number of labels must match number of files")
    valid = set(CLASSES)
    saved = []
    for f, lab in zip(files, label_list):
        if lab not in valid:
            raise HTTPException(400, f"Invalid label '{lab}'. Must be one of {valid}")
        dest = os.path.join(RETRAIN_DIR, lab)
        os.makedirs(dest, exist_ok=True)
        contents = await f.read()
        path = os.path.join(dest, f"{uuid.uuid4().hex}_{f.filename}")
        with open(path, "wb") as out:
            out.write(contents)
        saved.append({"label": lab, "filename": f.filename})
    return {"status": "saved", "count": len(saved), "files": saved}


@app.post("/retrain")
def trigger_retrain(epochs: int = 5):
    """
    Retrain / fine-tune the production Advanced CNN using:
      - original data/train images
      - new images in data/retrain_buffer
    Uses the custom pre-trained model (MobileNetV2 transfer learning) as starting point.
    """
    from src.model import fine_tune_keras_model
    from sklearn.preprocessing import LabelEncoder

    # 1. Load original training data (same preprocessing as notebook)
    train_dir = os.path.join(PROJECT_ROOT, "data", "train")
    X_orig, y_orig, _ = load_images_from_dir(train_dir, img_size=IMG_SIZE, max_per_class=300)

    # 2. Load buffer images
    X_buf, y_buf = [], []
    for lab in CLASSES:
        d = os.path.join(RETRAIN_DIR, lab)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                img = Image.open(os.path.join(d, fname)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                X_buf.append(np.array(img, dtype=np.float32) / 255.0)
                y_buf.append(lab)
            except Exception:
                pass

    if len(X_buf) == 0:
        return {
            "status": "skipped",
            "message": "No new data in retrain_buffer. Upload images first via /upload_retrain.",
        }

    # 3. Combine
    if len(X_orig) > 0:
        X_all = np.concatenate([X_orig, np.array(X_buf)], axis=0)
        y_all = np.concatenate([y_orig, np.array(y_buf)], axis=0)
    else:
        X_all = np.array(X_buf)
        y_all = np.array(y_buf)

    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)  # keep consistent order
    y_enc = le.transform(y_all)

    # 4. Load existing best model and fine-tune (custom pre-trained model)
    model = load_keras_model_compat(MODEL_PATH, compile=False)
    model, history = fine_tune_keras_model(
        model, X_all, y_enc, epochs=epochs, batch_size=16, learning_rate=1e-4
    )

    # 5. Persist
    model.save(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    reload_model()

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_original": int(len(X_orig)),
        "n_new": int(len(X_buf)),
        "n_total": int(len(X_all)),
        "epochs": epochs,
        "model_type": "advanced_cnn_finetune",
    }
    RETRAIN_LOG.append(entry)
    return {
        "status": "success",
        "message": "Advanced CNN fine-tuned on original + new data and reloaded",
        "details": entry,
    }


@app.get("/metrics")
def metrics():
    p = os.path.join(PROJECT_ROOT, "models", "metrics_summary.pkl")
    if not os.path.exists(p):
        return {"message": "No metrics yet"}
    return joblib.load(p)


@app.get("/dataset_stats")
def dataset_stats():
    stats = {}
    for split in ["train", "test"]:
        stats[split] = {}
        for c in CLASSES:
            d = os.path.join(PROJECT_ROOT, "data", split, c)
            stats[split][c] = (
                len([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                if os.path.isdir(d)
                else 0
            )
    buf = {}
    for c in CLASSES:
        d = os.path.join(RETRAIN_DIR, c)
        buf[c] = len(os.listdir(d)) if os.path.isdir(d) else 0
    stats["retrain_buffer"] = buf
    return stats


@app.get("/retrain_log")
def retrain_log():
    return {"log": RETRAIN_LOG}