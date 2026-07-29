"""
Model training & evaluation for Diabetic Retinopathy.
Includes classical ML (for notebook experiments) and CNN / fine-tuning path
used by the production API (best model = Advanced CNN with MobileNetV2).
"""
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def train_random_forest(X, y, n_estimators=200, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=-1, class_weight="balanced")
    model.fit(X, y)
    return model


def train_logistic(X, y, C=1.0, max_iter=2000, random_state=42):
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state,
                               solver="lbfgs", class_weight="balanced")
    model.fit(X, y)
    return model


def train_regularized_logistic(X, y, C=0.1, max_iter=5000, random_state=42):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state,
            solver="lbfgs", class_weight="balanced",
        ),
    )
    model.fit(X, y)
    return model


def train_svm(X, y, C=1.0, random_state=42):
    model = SVC(C=C, probability=True, random_state=random_state, class_weight="balanced")
    model.fit(X, y)
    return model


def train_xgboost(X, y, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                          max_depth=max_depth, random_state=random_state,
                          eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X, y)
    return model


def build_cnn_model(input_shape=(128, 128, 3), num_classes=5):
    """Simple CNN – images are ALREADY in [0,1], do NOT rescale again."""
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_advanced_cnn(input_shape=(128, 128, 3), num_classes=5):
    """
    MobileNetV2 transfer-learning model – fully compatible with TF 2.13.
    Images must already be in [0, 1].
    """
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze backbone first

    inputs = keras.Input(shape=input_shape)
    # light augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    # MobileNetV2 expects [-1, 1]
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def train_cnn_model(X_train, y_train, X_val=None, y_val=None, input_shape=(128, 128, 3),
                    epochs=20, batch_size=32, advanced=True):
    """Train either the simple CNN or the Advanced (MobileNetV2) CNN."""
    tf.keras.backend.clear_session()
    num_classes = len(np.unique(y_train))

    if advanced:
        model = build_advanced_cnn(input_shape=input_shape, num_classes=num_classes)
        lr = 1e-3
    else:
        model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
        lr = 1e-3

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6, monitor="val_loss"),
    ]

    class_weights = None
    if len(np.unique(y_train)) > 1:
        class_weights = dict(enumerate(
            compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))

    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            class_weight=class_weights, verbose=1,
        )
    else:
        history = model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, class_weight=class_weights, verbose=1,
        )
    return model, history


def fine_tune_keras_model(model, X_train, y_train, epochs=5, batch_size=16, learning_rate=1e-4):
    """
    Fine-tune an existing Keras model (the production Advanced CNN) on new + original data.
    """
    for layer in model.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    class_weights = None
    if len(np.unique(y_train)) > 1:
        class_weights = dict(enumerate(
            compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def evaluate_model(model, X_test, y_test, class_names=None):
    if hasattr(model, "predict_proba"):
        preds = model.predict(X_test)
        try:
            probs = model.predict_proba(X_test)
        except Exception:
            probs = None
    else:
        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
    }
    if probs is not None and len(np.unique(y_test)) > 2:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc_ovr"] = None
    report = classification_report(y_test, preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    return metrics, preds, probs, report, cm


def plot_cm(cm, class_names, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if hasattr(model, "save") and callable(getattr(model, "save")) and not hasattr(model, "predict_proba"):
        model.save(path)
    else:
        joblib.dump(model, path)
    print(f"Saved → {path}")


def load_model(path):
    from .keras_compat import load_keras_model_compat
    if path.endswith((".keras", ".h5", ".tf")):
        return load_keras_model_compat(path, compile=False)
    return joblib.load(path)