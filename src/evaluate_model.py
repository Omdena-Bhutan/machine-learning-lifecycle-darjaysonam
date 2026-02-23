from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main() -> None:
    params = load_params()
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Paths
    raw_dir = Path(params["data"]["raw_dir"])
    test_path = raw_dir / params["data"]["test_file"]

    model_dir = Path("models/trained/distilbert_sentiment")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}. Run: dvc repro train")

    # Load data
    test_df = pd.read_parquet(test_path)
    X_test = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    # Load HF model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference (batching avoids memory spikes)
    batch_size = int(params.get("train", {}).get("batch_size", 16))
    probs_pos = []
    preds = []

    with torch.inference_mode():
        for batch_texts in batched(X_test, batch_size):
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=int(params.get("train", {}).get("max_length", 128)),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits  # [batch, 2]
            prob = torch.softmax(logits, dim=-1)[:, 1]  # positive class prob
            pred = (prob >= 0.5).long()

            probs_pos.extend(prob.detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())

    y_pred = np.array(preds, dtype=int)
    y_proba = np.array(probs_pos, dtype=float)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name="evaluate_transformer"):
        mlflow.log_metrics({"test_accuracy": acc, "test_f1": f1, "test_roc_auc": auc})

        # Confusion matrix
        fig1, ax1 = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax1)
        cm_path = reports_dir / "confusion_matrix.png"
        fig1.savefig(cm_path, bbox_inches="tight")
        plt.close(fig1)
        mlflow.log_artifact(str(cm_path), artifact_path="reports")

        # ROC curve
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax2)
        roc_path = reports_dir / "roc_curve.png"
        fig2.savefig(roc_path, bbox_inches="tight")
        plt.close(fig2)
        mlflow.log_artifact(str(roc_path), artifact_path="reports")

        print({"test_accuracy": acc, "test_f1": f1, "test_roc_auc": auc})
        print("Saved:", cm_path, roc_path)


if __name__ == "__main__":
    main()
