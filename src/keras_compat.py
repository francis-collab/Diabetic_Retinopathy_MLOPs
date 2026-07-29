"""Thin wrapper – model is now saved with the same TF version used for serving."""
import os
import tensorflow as tf


def load_keras_model_compat(model_path, compile=False):
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return tf.keras.models.load_model(model_path, compile=compile)