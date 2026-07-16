import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.data.normalize_annotations import normalize


class NormalizeAnnotationTests(unittest.TestCase):
    def test_normalize_resizes_image_and_scales_xyxy(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            annotation_dir = tmp_path / "annotations"
            image_root = tmp_path / "images"
            annotation_dir.mkdir()
            image_root.mkdir()
            sid = "1" * 32
            fid = "2" * 32
            image_name = f"{sid}_{fid}.png"
            Image.new("RGB", (100, 50), color=(10, 20, 30)).save(image_root / image_name)
            annotation_name = f"{sid}_{fid}.json"
            (annotation_dir / annotation_name).write_text(
                json.dumps({"image": image_name, "bbox": [[10, 5, 50, 25]]}), encoding="utf-8"
            )
            manifest = tmp_path / "manifest.txt"
            manifest.write_text(annotation_name + "\n", encoding="utf-8")
            output = tmp_path / "normalized.jsonl"

            count = normalize(
                annotation_dir=annotation_dir,
                annotation_manifest=manifest,
                image_root=image_root,
                prepared_image_dir=tmp_path / "prepared",
                output_jsonl=output,
                boxes_key=None,
                image_key=None,
                box_format="xyxy",
                default_extension=".jpg",
                canvas=840,
                problem="target",
                jpeg_quality=90,
            )

            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(count, 1)
            self.assertEqual(row["boxes"], [[84, 84, 420, 420]])
            with Image.open(row["image"]) as image:
                self.assertEqual(image.size, (840, 840))


if __name__ == "__main__":
    unittest.main()
