from pathlib import Path
import pandas as pd
import yaml
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
import joblib  # scikit-learn uses this internally; safe for local load

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Paths
    raw_dir = Path(params["data"]["raw_dir"])
    test_path = raw_dir / params["data"]["test_file"]

    model_path = Path("models/trained/baseline_tfidf_logreg")  # your DVC-tracked model dir
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_df = pd.read_parquet(test_path)
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(int)

    # Load sklearn model saved by MLflow (local folder)
    model = joblib.load(model_path / "model.pkl")

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name="evaluate_baseline"):
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
