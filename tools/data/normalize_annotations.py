#!/usr/bin/env python3
"""Normalize per-frame JSON annotations and resize images to a fixed square canvas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


BOX_KEYS = ("bbox", "boxes", "bboxes", "annotations")
IMAGE_KEYS = ("image", "image_path", "img_name", "imagePath", "filename")


def nested_get(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(dotted_key)
    return current


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    raise KeyError(f"none of {keys!r} found")


def parse_boxes(raw: Any, box_format: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise TypeError(f"boxes must be a list, got {type(raw).__name__}")
    boxes: list[list[float]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            item = item.get("bbox", item.get("bbox_2d", item.get("box")))
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(f"invalid box at index {index}: {item!r}")
        x1, y1, a, b = [float(value) for value in item]
        if box_format == "xywh":
            x2, y2 = x1 + a, y1 + b
        else:
            x2, y2 = a, b
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate box at index {index}: {item!r}")
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        raise ValueError("annotation contains no boxes")
    return boxes


def resolve_image_path(
    record: dict[str, Any], annotation_path: Path, image_root: Path, image_key: str | None, default_extension: str
) -> Path:
    image_ref: str | None = None
    if image_key:
        image_ref = str(nested_get(record, image_key))
    else:
        for key in IMAGE_KEYS:
            if key in record and record[key]:
                image_ref = str(record[key])
                break
    if image_ref is None:
        image_ref = annotation_path.stem + default_extension
    path = Path(image_ref)
    return path if path.is_absolute() else image_root / path


def scale_box(box: list[float], width: int, height: int, canvas: int) -> list[int]:
    x1, y1, x2, y2 = box
    scaled = [
        round(max(0.0, min(x1, width)) * canvas / width),
        round(max(0.0, min(y1, height)) * canvas / height),
        round(max(0.0, min(x2, width)) * canvas / width),
        round(max(0.0, min(y2, height)) * canvas / height),
    ]
    scaled[0] = min(scaled[0], canvas - 1)
    scaled[1] = min(scaled[1], canvas - 1)
    scaled[2] = max(scaled[0] + 1, min(scaled[2], canvas))
    scaled[3] = max(scaled[1] + 1, min(scaled[3], canvas))
    return scaled


def normalize(
    annotation_dir: Path,
    annotation_manifest: Path,
    image_root: Path,
    prepared_image_dir: Path,
    output_jsonl: Path,
    boxes_key: str | None,
    image_key: str | None,
    box_format: str,
    default_extension: str,
    canvas: int,
    problem: str,
    jpeg_quality: int,
) -> int:
    prepared_image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = [line.strip() for line in annotation_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = 0

    with output_jsonl.open("w", encoding="utf-8") as output:
        for line_no, annotation_ref in enumerate(manifest_rows, 1):
            annotation_path = Path(annotation_ref)
            if not annotation_path.is_absolute():
                annotation_path = annotation_dir / annotation_path
            try:
                record = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
                if not isinstance(record, dict):
                    raise TypeError("top-level JSON value must be an object")
                raw_boxes = nested_get(record, boxes_key) if boxes_key else first_present(record, BOX_KEYS)
                boxes = parse_boxes(raw_boxes, box_format)
                source_image = resolve_image_path(record, annotation_path, image_root, image_key, default_extension)
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)

                scene_id = annotation_path.stem.split("_", 1)[0].lower()
                relative_image = Path(scene_id) / f"{annotation_path.stem}.jpg"
                prepared_image = prepared_image_dir / relative_image
                prepared_image.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source_image) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    scaled_boxes = [scale_box(box, width, height, canvas) for box in boxes]
                    image.resize((canvas, canvas), Image.Resampling.BILINEAR).save(
                        prepared_image, format="JPEG", quality=jpeg_quality, optimize=True
                    )

                normalized = {
                    "id": annotation_path.stem,
                    "scene_id": scene_id,
                    "image": str(prepared_image.resolve()),
                    "img_name": relative_image.as_posix(),
                    "boxes": scaled_boxes,
                    "bbox": scaled_boxes,
                    "problem": problem,
                    "width": canvas,
                    "height": canvas,
                    "source_width": width,
                    "source_height": height,
                }
                output.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            except Exception as exc:
                raise RuntimeError(f"Failed manifest line {line_no}, annotation {annotation_path}: {exc}") from exc
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prepared-image-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--boxes-key", help="Optional dotted key, e.g. annotations.boxes")
    parser.add_argument("--image-key", help="Optional dotted key for the image filename")
    parser.add_argument("--box-format", choices=("xyxy", "xywh"), default="xyxy")
    parser.add_argument("--default-extension", default=".jpg")
    parser.add_argument("--canvas", type=int, default=840)
    parser.add_argument("--problem", default="target object")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = normalize(**vars(args))
    print(json.dumps({"samples": count, "output": str(args.output_jsonl)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
