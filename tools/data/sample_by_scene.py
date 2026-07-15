#!/usr/bin/env python3
"""Deterministically sample records in scene-round-robin order for broad coverage."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sample_by_scene(records: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    if samples > len(records):
        raise ValueError(f"requested {samples} records but only {len(records)} are available")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id:
            raise ValueError(f"record {record.get('id')!r} has no scene_id")
        groups[scene_id].append(record)
    for group in groups.values():
        rng.shuffle(group)
    scene_ids = sorted(groups)
    rng.shuffle(scene_ids)

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < samples:
        added_this_round = 0
        for scene_id in scene_ids:
            group = groups[scene_id]
            if round_index < len(group):
                selected.append(group[round_index])
                added_this_round += 1
                if len(selected) == samples:
                    break
        if added_this_round == 0:
            raise AssertionError("sampling exhausted unexpectedly")
        round_index += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20250715)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sample_by_scene(records, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"selected": len(selected), "scenes": len({row['scene_id'] for row in selected})}))


if __name__ == "__main__":
    main()
