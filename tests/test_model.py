from pathlib import Path

def test_model_artifact_exists():
    p = Path("models/trained/baseline_tfidf_logreg/model.pkl")
    assert p.exists(), "Run: dvc repro train"
