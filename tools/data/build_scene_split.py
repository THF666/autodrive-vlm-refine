#!/usr/bin/env python3
"""Create a deterministic 90/10 split without leaking frames across scenes."""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


SCENE_LINE_RE = re.compile(r"^([0-9a-fA-F]{32})\s+(\d+)\s*$")
ANNOTATION_RE = re.compile(r"^(?P<scene>[0-9a-fA-F]{32})_(?P<frame>[0-9a-fA-F]{32})\.json$")
TOTAL_SCENES_RE = re.compile(r"^总场景数\s*[:：]\s*(\d+)\s*$")
TOTAL_IMAGES_RE = re.compile(r"^总图像数\s*[:：]\s*(\d+)\s*$")


@dataclass(frozen=True)
class SceneStat:
    scene_id: str
    image_count: int


def parse_scene_stats(path: Path) -> list[SceneStat]:
    """Parse the UUID/count lines and validate the optional Chinese summaries."""
    scenes: list[SceneStat] = []
    summary_scenes: int | None = None
    summary_images: int | None = None

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            match = SCENE_LINE_RE.fullmatch(line)
            if match:
                count = int(match.group(2))
                if count <= 0:
                    raise ValueError(f"{path}:{line_no}: image count must be positive")
                scenes.append(SceneStat(match.group(1).lower(), count))
                continue
            match = TOTAL_SCENES_RE.fullmatch(line)
            if match:
                summary_scenes = int(match.group(1))
                continue
            match = TOTAL_IMAGES_RE.fullmatch(line)
            if match:
                summary_images = int(match.group(1))
                continue
            raise ValueError(f"{path}:{line_no}: unrecognized line: {line!r}")

    if not scenes:
        raise ValueError(f"No scene rows found in {path}")
    duplicates = [scene_id for scene_id, count in Counter(s.scene_id for s in scenes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate scene ids: {duplicates[:5]}")
    if summary_scenes is not None and summary_scenes != len(scenes):
        raise ValueError(f"Scene summary says {summary_scenes}, parsed {len(scenes)}")
    parsed_images = sum(scene.image_count for scene in scenes)
    if summary_images is not None and summary_images != parsed_images:
        raise ValueError(f"Image summary says {summary_images}, parsed {parsed_images}")
    return scenes


def discover_annotations(annotation_dir: Path, scenes: list[SceneStat]) -> dict[str, list[Path]]:
    """Index annotation JSON files and fail if their per-scene counts disagree."""
    expected = {scene.scene_id: scene.image_count for scene in scenes}
    found: dict[str, list[Path]] = {scene.scene_id: [] for scene in scenes}
    invalid_names: list[str] = []
    unknown_scenes: list[str] = []

    for path in sorted(annotation_dir.rglob("*.json")):
        match = ANNOTATION_RE.fullmatch(path.name)
        if not match:
            invalid_names.append(str(path.relative_to(annotation_dir)))
            continue
        scene_id = match.group("scene").lower()
        if scene_id not in expected:
            unknown_scenes.append(scene_id)
            continue
        found[scene_id].append(path)

    if invalid_names:
        raise ValueError(f"JSON files with unexpected names (first 5): {invalid_names[:5]}")
    if unknown_scenes:
        raise ValueError(f"Annotation files reference unknown scenes (first 5): {sorted(set(unknown_scenes))[:5]}")

    mismatches = [
        (scene_id, expected[scene_id], len(paths))
        for scene_id, paths in found.items()
        if expected[scene_id] != len(paths)
    ]
    if mismatches:
        preview = ", ".join(f"{scene}: expected={expected_count}, actual={actual}" for scene, expected_count, actual in mismatches[:8])
        raise ValueError(f"Per-scene annotation counts do not match: {preview}")
    return found


def _local_search(selected_ids: set[str], counts: dict[str, int], target_images: int) -> set[str]:
    """Improve image-count balance while preserving the exact number of scenes."""
    selected = set(selected_ids)
    current_images = sum(counts[scene_id] for scene_id in selected)

    while True:
        unselected = sorted((count, scene_id) for scene_id, count in counts.items() if scene_id not in selected)
        unselected_counts = [item[0] for item in unselected]
        current_error = abs(current_images - target_images)
        best: tuple[int, str, str, int] | None = None

        for old_id in sorted(selected):
            desired_new_count = target_images - (current_images - counts[old_id])
            pos = bisect.bisect_left(unselected_counts, desired_new_count)
            for idx in range(max(0, pos - 2), min(len(unselected), pos + 3)):
                new_count, new_id = unselected[idx]
                candidate_images = current_images - counts[old_id] + new_count
                candidate_error = abs(candidate_images - target_images)
                candidate = (candidate_error, old_id, new_id, candidate_images)
                if candidate_error < current_error and (best is None or candidate < best):
                    best = candidate

        if best is None:
            return selected
        _, old_id, new_id, current_images = best
        selected.remove(old_id)
        selected.add(new_id)


def choose_test_scenes(
    scenes: list[SceneStat], test_ratio: float = 0.10, seed: int = 20250715, restarts: int = 32
) -> tuple[set[str], int, int]:
    """Choose whole scenes close to both the requested scene and image ratios."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    if restarts <= 0:
        raise ValueError("restarts must be positive")

    target_scene_count = round(len(scenes) * test_ratio)
    target_image_count = round(sum(scene.image_count for scene in scenes) * test_ratio)
    counts = {scene.scene_id: scene.image_count for scene in scenes}
    scene_ids = sorted(counts)
    rng = random.Random(seed)
    best: tuple[int, tuple[str, ...]] | None = None

    for _ in range(restarts):
        initial = set(rng.sample(scene_ids, target_scene_count))
        candidate = _local_search(initial, counts, target_image_count)
        ordered = tuple(sorted(candidate))
        score = (abs(sum(counts[scene_id] for scene_id in candidate) - target_image_count), ordered)
        if best is None or score < best:
            best = score

    assert best is not None
    return set(best[1]), target_scene_count, target_image_count


def write_split(
    scenes: list[SceneStat],
    annotation_dir: Path,
    annotations: dict[str, list[Path]],
    output_dir: Path,
    test_scene_ids: set[str],
    target_scene_count: int,
    target_image_count: int,
    test_ratio: float,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {scene.scene_id: scene.image_count for scene in scenes}
    all_scene_ids = set(counts)
    train_scene_ids = all_scene_ids - test_scene_ids

    def relative_paths(scene_ids: set[str]) -> list[str]:
        return sorted(
            path.relative_to(annotation_dir).as_posix()
            for scene_id in scene_ids
            for path in annotations[scene_id]
        )

    train_annotations = relative_paths(train_scene_ids)
    test_annotations = relative_paths(test_scene_ids)
    train_images = sum(counts[scene_id] for scene_id in train_scene_ids)
    test_images = sum(counts[scene_id] for scene_id in test_scene_ids)
    total_images = train_images + test_images

    files = {
        "train_scenes.txt": sorted(train_scene_ids),
        "test_scenes.txt": sorted(test_scene_ids),
        "train_annotations.txt": train_annotations,
        "test_annotations.txt": test_annotations,
    }
    for name, values in files.items():
        (output_dir / name).write_text("\n".join(values) + "\n", encoding="utf-8")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "strategy": "whole-scene split with fixed scene count and image-count local search",
        "seed": seed,
        "requested_test_ratio": test_ratio,
        "total": {"scenes": len(scenes), "images": total_images},
        "target_test": {"scenes": target_scene_count, "images": target_image_count},
        "actual_train": {
            "scenes": len(train_scene_ids),
            "images": train_images,
            "image_ratio": train_images / total_images,
        },
        "actual_test": {
            "scenes": len(test_scene_ids),
            "images": test_images,
            "image_ratio": test_images / total_images,
        },
        "scenes": [asdict(scene) | {"split": "test" if scene.scene_id in test_scene_ids else "train"} for scene in scenes],
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = (
        f"seed: {seed}\n"
        f"requested test ratio: {test_ratio:.2%}\n"
        f"total: {len(scenes)} scenes / {total_images} images\n"
        f"train: {len(train_scene_ids)} scenes / {train_images} images ({train_images / total_images:.4%})\n"
        f"test: {len(test_scene_ids)} scenes / {test_images} images ({test_images / total_images:.4%})\n"
        f"target test: {target_scene_count} scenes / {target_image_count} images\n"
        f"test image target error: {test_images - target_image_count:+d}\n"
    )
    (output_dir / "split_report.txt").write_text(report, encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-stats", type=Path, required=True, help="TXT with scene UUID and image count")
    parser.add_argument("--annotation-dir", type=Path, required=True, help="Root containing per-frame JSON files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20250715)
    parser.add_argument("--restarts", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenes = parse_scene_stats(args.scene_stats)
    annotations = discover_annotations(args.annotation_dir, scenes)
    test_ids, target_scenes, target_images = choose_test_scenes(
        scenes, test_ratio=args.test_ratio, seed=args.seed, restarts=args.restarts
    )
    manifest = write_split(
        scenes,
        args.annotation_dir,
        annotations,
        args.output_dir,
        test_ids,
        target_scenes,
        target_images,
        args.test_ratio,
        args.seed,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["actual_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
