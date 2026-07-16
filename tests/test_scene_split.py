from pathlib import Path
import tempfile
import unittest

from tools.data.build_scene_split import (
    choose_test_scenes,
    discover_annotations,
    parse_scene_stats,
    write_split,
)


def scene_id(index: int) -> str:
    return f"{index:032x}"


def frame_id(index: int) -> str:
    return f"{index + 10_000:032x}"


def make_dataset(tmp_path: Path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    counts = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    lines = []
    frame_index = 0
    for index, count in enumerate(counts):
        sid = scene_id(index)
        lines.append(f"{sid}\t{count}")
        for _ in range(count):
            (annotation_dir / f"{sid}_{frame_id(frame_index)}.json").write_text("{}", encoding="utf-8")
            frame_index += 1
    lines.extend([f"总场景数: {len(counts)}", f"总图像数: {sum(counts)}"])
    stats = tmp_path / "scene_stats.txt"
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats, annotation_dir, counts


class SceneSplitTests(unittest.TestCase):
    def test_scene_split_is_deterministic_and_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            stats_path, annotation_dir, counts = make_dataset(tmp_path)
            scenes = parse_scene_stats(stats_path)
            annotations = discover_annotations(annotation_dir, scenes)
            first, target_scenes, target_images = choose_test_scenes(scenes, test_ratio=0.10, seed=123, restarts=8)
            second, _, _ = choose_test_scenes(scenes, test_ratio=0.10, seed=123, restarts=8)

            self.assertEqual(first, second)
            self.assertEqual(len(first), target_scenes)
            self.assertEqual(target_scenes, 2)
            actual_test_images = sum(scene.image_count for scene in scenes if scene.scene_id in first)
            self.assertLessEqual(abs(actual_test_images - target_images), 1)

            output_dir = tmp_path / "split"
            manifest = write_split(
                scenes,
                annotation_dir,
                annotations,
                output_dir,
                first,
                target_scenes,
                target_images,
                0.10,
                123,
            )
            train_scenes = set((output_dir / "train_scenes.txt").read_text(encoding="utf-8").splitlines())
            test_scenes = set((output_dir / "test_scenes.txt").read_text(encoding="utf-8").splitlines())
            self.assertFalse(train_scenes & test_scenes)
            self.assertEqual(len(train_scenes | test_scenes), len(counts))
            self.assertEqual(manifest["actual_test"]["scenes"], 2)

    def test_annotation_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            stats_path, annotation_dir, _ = make_dataset(tmp_path)
            scenes = parse_scene_stats(stats_path)
            next(annotation_dir.glob("*.json")).unlink()
            with self.assertRaisesRegex(ValueError, "counts do not match"):
                discover_annotations(annotation_dir, scenes)


if __name__ == "__main__":
    unittest.main()
