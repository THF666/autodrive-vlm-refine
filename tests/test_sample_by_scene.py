import unittest

from tools.data.sample_by_scene import sample_by_scene


class SampleBySceneTests(unittest.TestCase):
    def test_first_round_covers_distinct_scenes(self):
        records = [
            {"id": f"{scene}-{frame}", "scene_id": scene}
            for scene in ("a", "b", "c")
            for frame in range(4)
        ]
        selected = sample_by_scene(records, samples=6, seed=7)
        first_round = {record["scene_id"] for record in selected[:3]}
        self.assertEqual(first_round, {"a", "b", "c"})
        self.assertEqual(len({record["id"] for record in selected}), 6)


if __name__ == "__main__":
    unittest.main()
