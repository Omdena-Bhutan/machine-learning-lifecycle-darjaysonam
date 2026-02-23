from pathlib import Path
import pytest

def test_raw_data_exists():
    train = Path("data/raw/imdb_train.parquet")
    test = Path("data/raw/imdb_test.parquet")

    if not (train.exists() and test.exists()):
        pytest.skip("DVC raw data not present in CI. Run `dvc repro get_data` locally or configure DVC remote for CI.")

    assert train.exists()
    assert test.exists()

