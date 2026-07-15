#!/usr/bin/env python3
"""Combine compatible HF datasets while keeping only Stage-1/3 contract columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


REQUIRED_COLUMNS = ("id", "problem", "solution", "image")


def load_train(source: str):
    path = Path(source)
    loaded = load_from_disk(str(path)) if path.exists() else load_dataset(source)
    if isinstance(loaded, DatasetDict):
        if "train" not in loaded:
            raise ValueError(f"{source} has no train split")
        loaded = loaded["train"]
    missing = set(REQUIRED_COLUMNS) - set(loaded.column_names)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    return loaded.select_columns(REQUIRED_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="save_to_disk paths or HF dataset IDs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250715)
    args = parser.parse_args()

    datasets = [load_train(source) for source in args.input]
    combined = concatenate_datasets(datasets).shuffle(seed=args.seed)
    DatasetDict({"train": combined}).save_to_disk(str(args.output_dir))
    print(json.dumps({"sources": len(datasets), "samples": len(combined), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
