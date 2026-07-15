#!/usr/bin/env python3
"""Build deterministic cold-start Refine SFT conversations from normalized boxes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


Box = list[int]
SCENARIOS = ("jitter", "missing", "false_positive", "duplicate", "mixed", "near_correct")
DEFAULT_WEIGHTS = (32, 18, 14, 10, 20, 6)
REFINE_SCHEMA = (
    '{"per_bbox":[{"id":"B1","action":{"bbox":[dleft,dtop,dright,dbottom],'
    '"point":[dX,dY]}},{"id":"B2","action":"delete"}],'
    '"new_items":[{"bbox_2d":[x1,y1,x2,y2],"point_2d":[px,py]}]}'
)


def center(box: Box) -> list[int]:
    return [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]


def clamp_box(box: list[int], canvas: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), canvas - 1))
    y1 = max(0, min(int(y1), canvas - 1))
    x2 = max(x1 + 1, min(int(x2), canvas))
    y2 = max(y1 + 1, min(int(y2), canvas))
    return [x1, y1, x2, y2]


def jitter_box(box: Box, rng: random.Random, canvas: int, strength: float = 0.16) -> Box:
    width = box[2] - box[0]
    height = box[3] - box[1]
    dx = max(3, round(width * strength))
    dy = max(3, round(height * strength))
    deltas = [rng.randint(-dx, dx), rng.randint(-dy, dy), rng.randint(-dx, dx), rng.randint(-dy, dy)]
    candidate = clamp_box([value + delta for value, delta in zip(box, deltas)], canvas)
    if candidate == box:
        candidate = clamp_box([box[0] + 1, box[1], box[2] + 1, box[3]], canvas)
    return candidate


def random_false_box(rng: random.Random, canvas: int) -> Box:
    width = rng.randint(max(8, canvas // 20), max(16, canvas // 4))
    height = rng.randint(max(8, canvas // 20), max(16, canvas // 4))
    x1 = rng.randint(0, canvas - width)
    y1 = rng.randint(0, canvas - height)
    return [x1, y1, x1 + width, y1 + height]


def normalize_record(record: dict[str, Any], line_no: int, canvas: int) -> dict[str, Any]:
    image = record.get("image")
    raw_boxes = record.get("boxes", record.get("bbox"))
    if not isinstance(image, str) or not image:
        raise ValueError(f"line {line_no}: missing image")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError(f"line {line_no}: boxes must be a non-empty list")
    boxes: list[Box] = []
    for item in raw_boxes:
        if isinstance(item, dict):
            item = item.get("bbox_2d", item.get("bbox"))
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError(f"line {line_no}: invalid box {item!r}")
        boxes.append(clamp_box([round(float(value)) for value in item], canvas))
    return {
        "id": str(record.get("id", line_no)),
        "scene_id": str(record.get("scene_id", "")),
        "image": image,
        "boxes": boxes,
        "problem": str(record.get("problem", "target object")),
    }


def load_records(paths: list[Path], canvas: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    records.append(normalize_record(json.loads(line), line_no, canvas))
    if not records:
        raise ValueError("No input records")
    return records


def build_proposal(
    boxes: list[Box], scenario: str, rng: random.Random, canvas: int
) -> tuple[list[dict[str, Any]], list[int | None]]:
    kept_indices = list(range(len(boxes)))
    proposals: list[Box]

    if scenario in {"missing", "mixed"} and len(kept_indices) > 0:
        kept_indices.remove(rng.choice(kept_indices))
    proposals = [jitter_box(boxes[index], rng, canvas) for index in kept_indices]
    mapping: list[int | None] = list(kept_indices)

    if scenario == "near_correct":
        proposals = [list(box) for box in boxes]
        mapping = list(range(len(boxes)))
    if scenario in {"false_positive", "mixed"}:
        insert_at = rng.randint(0, len(proposals))
        proposals.insert(insert_at, random_false_box(rng, canvas))
        mapping.insert(insert_at, None)
    if scenario == "duplicate" and boxes:
        target = rng.randrange(len(boxes))
        proposals.append(jitter_box(boxes[target], rng, canvas, strength=0.08))
        mapping.append(None)

    proposal_items = [
        {"id": f"B{index + 1}", "bbox_2d": box, "point_2d": center(box)}
        for index, box in enumerate(proposals)
    ]
    return proposal_items, mapping


def build_refine_target(
    boxes: list[Box], proposal: list[dict[str, Any]], mapping: list[int | None]
) -> dict[str, Any]:
    per_bbox: list[dict[str, Any]] = []
    matched: set[int] = set()
    for item, gt_index in zip(proposal, mapping):
        if gt_index is None:
            per_bbox.append({"id": item["id"], "action": "delete"})
            continue
        matched.add(gt_index)
        source_box = item["bbox_2d"]
        target_box = boxes[gt_index]
        box_delta = [target - source for source, target in zip(source_box, target_box)]
        source_point = item["point_2d"]
        target_point = center(target_box)
        point_delta = [target - source for source, target in zip(source_point, target_point)]
        per_bbox.append({"id": item["id"], "action": {"bbox": box_delta, "point": point_delta}})

    new_items = [
        {"bbox_2d": box, "point_2d": center(box)} for index, box in enumerate(boxes) if index not in matched
    ]
    return {"per_bbox": per_bbox, "new_items": new_items}


def replay(proposal: list[dict[str, Any]], target: dict[str, Any]) -> list[Box]:
    actions = {item["id"]: item["action"] for item in target["per_bbox"]}
    output: list[Box] = []
    for item in proposal:
        action = actions[item["id"]]
        if action == "delete":
            continue
        output.append([value + delta for value, delta in zip(item["bbox_2d"], action["bbox"])])
    output.extend(item["bbox_2d"] for item in target["new_items"])
    return output


def describe_target(target: dict[str, Any]) -> str:
    corrections = sum(1 for item in target["per_bbox"] if item["action"] != "delete")
    deletions = sum(1 for item in target["per_bbox"] if item["action"] == "delete")
    additions = len(target["new_items"])
    return f"Check every candidate: correct {corrections}, delete {deletions}, and add {additions} missed object(s)."


def build_sample(record: dict[str, Any], scenario: str, rng: random.Random, canvas: int) -> dict[str, Any]:
    proposal, mapping = build_proposal(record["boxes"], scenario, rng, canvas)
    target = build_refine_target(record["boxes"], proposal, mapping)
    if sorted(replay(proposal, target)) != sorted(record["boxes"]):
        raise AssertionError(f"Refine target does not replay to ground truth for {record['id']}")
    proposal_json = json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    user_content = (
        f'<image>Locate "{record["problem"]}".\n'
        f"Your previous predictions are {proposal_json}.\n"
        f"Now refine them using this JSON shape: {REFINE_SCHEMA}"
    )
    assistant_content = f"<think>{describe_target(target)}</think><answer>{target_json}</answer>"
    return {
        "messages": [
            {"role": "system", "content": "You are a precise 2D object detection assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [record["image"]],
        "metadata": {
            "source_id": record["id"],
            "scene_id": record["scene_id"],
            "scenario": scenario,
            "proposal": proposal,
            "ground_truth": record["boxes"],
        },
    }


def generate(
    records: list[dict[str, Any]], sample_count: int, seed: int, canvas: int, scenario_weights: tuple[int, ...]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    order = list(range(len(records)))
    samples: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    for sample_index in range(sample_count):
        if sample_index % len(order) == 0:
            rng.shuffle(order)
        record = records[order[sample_index % len(order)]]
        scenario = rng.choices(SCENARIOS, weights=scenario_weights, k=1)[0]
        samples.append(build_sample(record, scenario, rng, canvas))
        scenario_counts[scenario] += 1
    return samples, scenario_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Normalized business/public JSONL files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20250715)
    parser.add_argument("--canvas", type=int, default=840)
    parser.add_argument(
        "--scenario-weights",
        type=int,
        nargs=len(SCENARIOS),
        default=DEFAULT_WEIGHTS,
        metavar=tuple(name.upper() for name in SCENARIOS),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_records(args.input, args.canvas)
    samples, scenario_counts = generate(records, args.samples, args.seed, args.canvas, tuple(args.scenario_weights))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {"input_records": len(records), "output_samples": len(samples), "scenarios": scenario_counts},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
