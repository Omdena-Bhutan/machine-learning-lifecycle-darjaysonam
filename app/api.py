from pathlib import Path
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_FILE = Path("../models/trained/baseline_tfidf_logreg/model.pkl")

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_FILE}. Run: dvc repro train"
            )
        _model = joblib.load(MODEL_FILE)
    return _model

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not text or not isinstance(text, str):
        return jsonify({"error": "Provide JSON with a string field: 'text'"}), 400

    model = get_model()
    proba_pos = float(model.predict_proba([text])[0, 1])
    pred = int(proba_pos >= 0.5)
    label = "positive" if pred == 1 else "negative"

    return jsonify({"label": label, "proba_positive": proba_pos})


BASE_DIR = Path(__file__).resolve().parent

# Local default (when running from app/)
LOCAL_MODEL = (BASE_DIR / ".." / "models" / "trained" / "baseline_tfidf_logreg" / "model.pkl").resolve()

# Docker default (we'll copy model to /models/...)
DOCKER_MODEL = Path("/models/trained/baseline_tfidf_logreg/model.pkl")

MODEL_FILE = DOCKER_MODEL if DOCKER_MODEL.exists() else LOCAL_MODEL

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
