import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use MODEL_DIR env var to override (handy for CI)
# Windows (PowerShell):  $env:MODEL_DIR="tests/assets/distilbert_sentiment"
# Linux/macOS:          export MODEL_DIR=tests/assets/distilbert_sentiment
DEFAULT_MODEL_DIR = "models/trained/distilbert_sentiment"
MODEL_DIR = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))

_tokenizer = None
_model = None
_device = None


def load_model():
    global _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None and _device is not None:
        return _tokenizer, _model, _device

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model folder not found at {MODEL_DIR}. Run: dvc repro train"
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)

    return _tokenizer, _model, _device


def predict_texts(texts: list[str], batch_size: int = 16, max_length: int = 128):
    tokenizer, model, device = load_model()

    probs_pos = []
    preds = []

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits  # [batch, 2]
            prob = torch.softmax(logits, dim=-1)[:, 1]  # positive class
            pred = (prob >= 0.5).long()

            probs_pos.extend(prob.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())

    return np.array(probs_pos, dtype=float), np.array(preds, dtype=int)


def predict_single(text: str, batch_size: int = 16, max_length: int = 128):
    probs, preds = predict_texts([text], batch_size=batch_size, max_length=max_length)
    proba_pos = float(probs[0])
    label = "positive" if int(preds[0]) == 1 else "negative"
    return {"label": label, "proba_positive": proba_pos}


def predict_batch(input_path: str, output_path: str, batch_size: int = 16, max_length: int = 128):
    inp = Path(input_path)
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Read input
    if inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)
    elif inp.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(inp)
    else:
        raise ValueError("Input must be .csv or .parquet")

    if "text" not in df.columns:
        raise ValueError("Input file must contain a 'text' column")

    texts = df["text"].astype(str).tolist()
    probs, preds = predict_texts(texts, batch_size=batch_size, max_length=max_length)

    df_out = df.copy()
    df_out["proba_positive"] = probs
    df_out["pred_label"] = pd.Series(preds).map({1: "positive", 0: "negative"})

    df_out.to_csv(outp, index=False)
    print(f"Saved predictions to: {outp}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("single", help="Predict sentiment for a single text")
    p1.add_argument("--text", required=True)
    p1.add_argument("--batch_size", type=int, default=16)
    p1.add_argument("--max_length", type=int, default=128)

    p2 = sub.add_parser("batch", help="Predict sentiment for a CSV/Parquet file")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--batch_size", type=int, default=16)
    p2.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()

    if args.cmd == "single":
        result = predict_single(args.text, batch_size=args.batch_size, max_length=args.max_length)
        print(result)
    else:
        predict_batch(args.input, args.output, batch_size=args.batch_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
