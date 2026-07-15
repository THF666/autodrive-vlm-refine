#!/usr/bin/env python3
"""Select the hardest training-only proposal results for Stage-3 GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL evaluation results joined with normalized GT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--score-key", default="metrics.f1")
    parser.add_argument("--hardest", choices=("lowest", "highest"), default="lowest")
    parser.add_argument("--allowed-scenes", type=Path, required=True, help="train_scenes.txt; blocks test leakage")
    parser.add_argument("--max-per-scene", type=int, default=0, help="0 disables the cap")
    args = parser.parse_args()

    allowed = {line.strip().lower() for line in args.allowed_scenes.read_text(encoding="utf-8").splitlines() if line.strip()}
    rows: list[tuple[float, str, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    with args.input.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            scene_id = str(record.get("scene_id", "")).lower()
            if scene_id not in allowed:
                raise ValueError(f"line {line_no}: scene {scene_id!r} is not in the training allow-list")
            record_id = str(record.get("id", record.get("image", line_no)))
            if record_id in seen_ids:
                raise ValueError(f"line {line_no}: duplicate record id {record_id!r}")
            seen_ids.add(record_id)
            score = float(nested_get(record, args.score_key))
            rows.append((score, scene_id, record_id, record))

    reverse = args.hardest == "highest"
    rows.sort(key=lambda item: (item[0], item[2]), reverse=reverse)
    selected: list[dict[str, Any]] = []
    per_scene: dict[str, int] = {}
    for _, scene_id, _, record in rows:
        if args.max_per_scene and per_scene.get(scene_id, 0) >= args.max_per_scene:
            continue
        selected.append(record)
        per_scene[scene_id] = per_scene.get(scene_id, 0) + 1
        if len(selected) == args.samples:
            break
    if len(selected) < args.samples:
        raise ValueError(f"Only {len(selected)} rows satisfy selection constraints; requested {args.samples}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"selected": len(selected), "scenes": len(per_scene), "output": str(args.output)}))


if __name__ == "__main__":
    main()
