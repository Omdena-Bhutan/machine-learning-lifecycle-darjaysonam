from pathlib import Path

def test_raw_data_exists():
    assert Path("data/raw/imdb_train.parquet").exists()
    assert Path("data/raw/imdb_test.parquet").exists()
