# src/data_loader.py
from pathlib import Path
import pandas as pd
import yaml
from datasets import load_dataset

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    raw_dir = Path(params["data"]["raw_dir"])
    train_file = params["data"]["train_file"]
    test_file = params["data"]["test_file"]
    dataset_name = params["data"]["dataset_name"]

    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name)  # imdb -> train/test
    pd.DataFrame(ds["train"]).to_parquet(raw_dir / train_file, index=False)
    pd.DataFrame(ds["test"]).to_parquet(raw_dir / test_file, index=False)

    print(f"Saved raw data to {raw_dir}")

if __name__ == "__main__":
    main()
