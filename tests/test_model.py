from pathlib import Path
import pytest


def test_model_artifact_exists():
    model_dir = Path("models/trained/distilbert_sentiment")

    if not model_dir.exists():
        pytest.skip(
            "Transformer model not present in CI. "
            "Run `dvc repro train` locally or configure DVC remote for CI."
        )

    assert model_dir.is_dir(), "Model path exists but is not a directory"

    # Required HuggingFace files
    assert (model_dir / "config.json").exists(), "Missing config.json"

    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )

    assert has_weights, "Missing model weight file (safetensors or pytorch_model.bin)"
