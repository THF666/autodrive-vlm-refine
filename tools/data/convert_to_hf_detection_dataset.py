#!/usr/bin/env python3
"""Convert normalized JSONL records to the on-disk HF format used by Seg-Zero/veRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, DatasetDict, Features, Image, Value


def center(box: list[int]) -> list[int]:
    return [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]


def normalize_box(raw_box: Any, canvas: int) -> list[int]:
    if isinstance(raw_box, dict):
        raw_box = raw_box.get("bbox_2d", raw_box.get("bbox"))
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError(f"invalid box: {raw_box!r}")
    x1, y1, x2, y2 = [round(float(value)) for value in raw_box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, y1 = max(0, min(x1, canvas - 1)), max(0, min(y1, canvas - 1))
    x2, y2 = max(x1 + 1, min(x2, canvas)), max(y1 + 1, min(y2, canvas))
    return [x1, y1, x2, y2]


def iter_rows(input_paths: list[Path], canvas: int, default_problem: str) -> Iterator[dict[str, Any]]:
    row_index = 0
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    image_path = Path(record["image"])
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    raw_boxes = record.get("boxes", record.get("bbox"))
                    if not isinstance(raw_boxes, list) or not raw_boxes:
                        raise ValueError("boxes must be a non-empty list")
                    boxes = [normalize_box(box, canvas) for box in raw_boxes]
                    solution = [{"bbox_2d": box, "point_2d": center(box)} for box in boxes]
                    yield {
                        "id": str(record.get("id", row_index)),
                        "problem": str(record.get("problem", default_problem)),
                        "solution": json.dumps(solution, ensure_ascii=False, separators=(",", ":")),
                        "image": str(image_path.resolve()),
                        "scene_id": str(record.get("scene_id", "")),
                    }
                    row_index += 1
                except Exception as exc:
                    raise RuntimeError(f"{input_path}:{line_no}: {exc}") from exc


def convert(input_paths: list[Path], output_dir: Path, canvas: int, default_problem: str) -> int:
    features = Features(
        {
            "id": Value("string"),
            "problem": Value("string"),
            "solution": Value("string"),
            "image": Image(),
            "scene_id": Value("string"),
        }
    )
    dataset = Dataset.from_generator(
        iter_rows,
        gen_kwargs={"input_paths": input_paths, "canvas": canvas, "default_problem": default_problem},
        features=features,
    )
    if len(dataset) == 0:
        raise ValueError("No dataset rows were generated")
    DatasetDict({"train": dataset}).save_to_disk(str(output_dir))
    return len(dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=840)
    parser.add_argument("--problem", default="target object")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = convert(args.input, args.output_dir, args.canvas, args.problem)
    print(json.dumps({"samples": count, "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
