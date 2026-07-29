import os
import json
import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_images_from_dir, CLASSES, IMG_SIZE
from src.model import train_cnn_model

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print(f"IMG_SIZE={IMG_SIZE} (must match notebook)")
    # Use more images for better quality (CPU-friendly limits)
    X_train_raw, y_train_raw, _ = load_images_from_dir(TRAIN_DIR, max_per_class=350)
    X_test_raw, y_test_raw, _ = load_images_from_dir(TEST_DIR, max_per_class=120)

    if len(X_train_raw) == 0:
        print("No training images found under data/train/. Aborting.")
        return

    print(f"Loaded train={len(X_train_raw)}  test={len(X_test_raw)}")

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_raw, y_train_raw, test_size=0.15, random_state=42, stratify=y_train_raw
    )

    le = LabelEncoder()
    le.fit(CLASSES)          # force consistent order
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)

    print("Training Advanced CNN (MobileNetV2 transfer learning) …")
    model, history = train_cnn_model(
        X_train_raw, y_train,
        X_val=X_val_raw, y_val=y_val,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        epochs=12,          # enough for transfer learning on CPU
        batch_size=32,
        advanced=True,      # ← MobileNetV2
    )

    test_probs = model.predict(X_test_raw, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "f1_weighted": float(f1_score(y_test, test_preds, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, test_preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, test_preds, average="weighted", zero_division=0)),
    }
    report = classification_report(y_test, test_preds, target_names=CLASSES, zero_division=0)

    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    model.save(model_path)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    summary = {
        "best_model": "AdvancedCNN_MobileNetV2",
        "best_metrics": metrics,
        "experiments": {"AdvancedCNN": {"metrics": metrics, "report": report}},
    }
    joblib.dump(summary, os.path.join(MODEL_DIR, "metrics_summary.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n===== Training complete =====")
    print(json.dumps(summary, indent=2))
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    main()