import os
from pathlib import Path
import argparse
import pandas as pd
import joblib

# MODEL_DIR = Path("models/trained/baseline_tfidf_logreg")
# MODEL_FILE = MODEL_DIR / "model.pkl"

MODEL_FILE = Path(os.getenv("MODEL_FILE", "models/trained/baseline_tfidf_logreg/model.pkl"))
MODEL_DIR = MODEL_FILE.parent

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_FILE}. Run: dvc repro train"
        )
    return joblib.load(MODEL_FILE)

def predict_single(text: str):
    model = load_model()
    proba_pos = float(model.predict_proba([text])[0, 1])
    pred = int(proba_pos >= 0.5)
    label = "positive" if pred == 1 else "negative"
    return {"label": label, "proba_positive": proba_pos}

def predict_batch(input_path: str, output_path: str):
    model = load_model()
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

    probs = model.predict_proba(df["text"].astype(str))[:, 1]
    preds = (probs >= 0.5).astype(int)

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

    p2 = sub.add_parser("batch", help="Predict sentiment for a CSV/Parquet file")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "single":
        result = predict_single(args.text)
        print(result)
    else:
        predict_batch(args.input, args.output)

if __name__ == "__main__":
    main()
