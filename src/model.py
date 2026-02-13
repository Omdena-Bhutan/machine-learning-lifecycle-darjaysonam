from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import yaml
import mlflow

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    params = load_params()
    t = params["train"]
    d = params["data"]

    seed = int(t.get("seed", 42))
    set_seed(seed)

    raw_dir = Path(d["raw_dir"])
    train_path = raw_dir / d["train_file"]
    test_path = raw_dir / d["test_file"]

    model_dir = Path("models/trained/distilbert_sentiment")
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Ensure correct types
    train_df["text"] = train_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    # Split train into train/val
    val_frac = float(t["validation_split"])
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(train_df) * val_frac)
    val_df = train_df.iloc[:n_val]
    tr_df = train_df.iloc[n_val:]

    ds_train = Dataset.from_pandas(tr_df[["text", "label"]], preserve_index=False)
    ds_val = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(t["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=int(t["max_length"]),
        )

    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    ds_train.set_format(type="torch", columns=cols)
    ds_val.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        t["model_name"], num_labels=2
    )

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"]
        return {"accuracy": acc, "f1": f1}

    args = TrainingArguments(
        output_dir=str(model_dir / "_hf_runs"),
        num_train_epochs=float(t["epochs"]),
        per_device_train_batch_size=int(t["batch_size"]),
        per_device_eval_batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="distilbert_transfer_learning"):
        mlflow.log_params(
            {
                "model_name": t["model_name"],
                "batch_size": t["batch_size"],
                "epochs": t["epochs"],
                "learning_rate": t["learning_rate"],
                "max_length": t["max_length"],
                "validation_split": t["validation_split"],
                "seed": seed,
            }
        )

        trainer.train()
        val_metrics = trainer.evaluate()

        mlflow.log_metrics(
            {
                "val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
                "val_f1": float(val_metrics.get("eval_f1", 0.0)),
                "val_loss": float(val_metrics.get("eval_loss", 0.0)),
            }
        )

        # Save trained model/tokenizer (production artifact)
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        print("Saved model to:", model_dir)
        print("Validation metrics:", val_metrics)


if __name__ == "__main__":
    main()
