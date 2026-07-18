"""
Cat vs Dog Classifier — Flask API
----------------------------------
POST /predict   multipart/form-data, field "image"
                → { success, label, confidence, emoji, color }
GET  /          → UI
GET  /health    → readiness check
"""
import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

ARTIFACT = Path(__file__).resolve().parent / "cat_dog_classify.keras"
IMG_SIZE = (160, 160)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
model = None


def load_model():
    global model
    if not ARTIFACT.exists():
        print(f"[warn] model not found at {ARTIFACT} — run train_model.py first")
        return
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(ARTIFACT)
        print(f"[ok] loaded {ARTIFACT.name}")
    except Exception as exc:
        print(f"[err] could not load model: {exc}")


def preprocess(file_bytes: bytes) -> np.ndarray:
    """Open image bytes, resize to 160×160, return float32 array (1,160,160,3)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # [0, 255]
    return np.expand_dims(arr, axis=0)       # (1,160,160,3)


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/health")
def health():
    return jsonify(model_loaded=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(success=False,
                       error="Model not loaded — run train_model.py first."), 503

    if "image" not in request.files:
        return jsonify(success=False, error="No file uploaded. Send a field named 'image'."), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify(success=False, error="Empty filename."), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(success=False,
                       error=f"Unsupported file type '{ext}'. Use JPEG, PNG, or WebP."), 415

    raw = f.read()
    if len(raw) > MAX_BYTES:
        return jsonify(success=False, error="File too large (max 10 MB)."), 413

    try:
        x = preprocess(raw)
    except Exception:
        return jsonify(success=False, error="Could not decode image. Is the file corrupted?"), 400

    # Model outputs a single sigmoid value: ~0 = cat, ~1 = dog
    # (matches the label_mode='binary' + alphabetical class ordering used during training)
    prob = float(model.predict(x, verbose=0)[0][0])
    is_dog = prob >= 0.5
    confidence = prob if is_dog else 1.0 - prob

    label = "Dog" if is_dog else "Cat"
    emoji = "🐶" if is_dog else "🐱"
    color = "dog" if is_dog else "cat"   # UI uses this to shift palette

    return jsonify(
        success=True,
        label=label,
        emoji=emoji,
        color=color,
        confidence=round(confidence * 100, 1),
        raw_prob=round(prob, 6),
    )


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
