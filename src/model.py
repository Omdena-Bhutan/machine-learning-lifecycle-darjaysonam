from pathlib import Path
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import shutil

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()

    raw_dir = Path(params["data"]["raw_dir"])
    train_path = raw_dir / params["data"]["train_file"]
    test_path  = raw_dir / params["data"]["test_file"]

    model_dir = Path("models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Load data
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(int)
    X_test, y_test   = test_df["text"].astype(str),  test_df["label"].astype(int)

    # Baseline model: TF-IDF + Logistic Regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=params["train"]["tfidf_max_features"],
            ngram_range=tuple(params["train"]["tfidf_ngram_range"]),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=params["train"]["logreg_C"],
            max_iter=params["train"]["logreg_max_iter"],
            solver="liblinear"
        ))
    ])

    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        # Log params
        mlflow.log_params({
            "tfidf_max_features": params["train"]["tfidf_max_features"],
            "tfidf_ngram_range": params["train"]["tfidf_ngram_range"],
            "logreg_C": params["train"]["logreg_C"],
            "logreg_max_iter": params["train"]["logreg_max_iter"],
        })

        # Train
        pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})

        # Save model locally + log to MLflow
        #local_model_path = model_dir / "baseline_tfidf_logreg"
        #mlflow.sklearn.save_model(pipe, str(local_model_path))
        #mlflow.sklearn.log_model(pipe, artifact_path="model")
        
        local_model_path = model_dir / "baseline_tfidf_logreg"
        if local_model_path.exists():
            shutil.rmtree(local_model_path)  # remove old model folder safely

        mlflow.sklearn.save_model(pipe, str(local_model_path))

        print("Saved model to:", local_model_path)
        print("Metrics:", {"accuracy": acc, "f1": f1, "roc_auc": auc})

if __name__ == "__main__":
    main()
