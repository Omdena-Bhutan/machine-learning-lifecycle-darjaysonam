from __future__ import annotations

from pathlib import Path
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

LOCAL_MODEL_DIR = (BASE_DIR / ".." / "models" / "trained" / "distilbert_sentiment").resolve()
DOCKER_MODEL_DIR = Path("/models/trained/distilbert_sentiment")

MODEL_DIR = DOCKER_MODEL_DIR if DOCKER_MODEL_DIR.exists() else LOCAL_MODEL_DIR

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_DIR}. Run: dvc repro train")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Provide JSON with a string field: 'text'"}), 400

    tokenizer, model = get_model()

    with torch.inference_mode():
        enc = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors="pt")
        logits = model(**enc).logits
        proba_pos = float(torch.softmax(logits, dim=-1)[0, 1].cpu().numpy())

    label = "positive" if proba_pos >= 0.5 else "negative"
    return jsonify({"label": label, "proba_positive": proba_pos})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
