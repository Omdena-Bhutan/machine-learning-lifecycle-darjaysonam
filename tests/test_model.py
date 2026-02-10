from pathlib import Path
import pytest

def test_model_artifact_exists():
    p = Path("models/trained/baseline_tfidf_logreg/model.pkl")

    if not p.exists():
        pytest.skip("Trained model not present in CI. Run `dvc repro train` locally or configure DVC remote for CI.")

    assert p.exists()

