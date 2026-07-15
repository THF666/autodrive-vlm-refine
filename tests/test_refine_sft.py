import random
import unittest

from tools.data.build_refine_sft import SCENARIOS, build_sample, generate


RECORD = {
    "id": "synthetic-frame",
    "scene_id": "0" * 32,
    "image": "/synthetic/not-opened.jpg",
    "boxes": [[100, 120, 220, 280], [400, 300, 520, 460]],
    "problem": "target object",
}


class RefineSftTests(unittest.TestCase):
    def test_every_error_scenario_replays_to_ground_truth(self):
        for index, scenario in enumerate(SCENARIOS):
            sample = build_sample(RECORD, scenario, random.Random(index), canvas=840)
            self.assertTrue(sample["messages"][1]["content"].startswith('<image>Locate "target object"'))
            self.assertTrue(sample["messages"][2]["content"].startswith("<think>"))
            self.assertEqual(sample["metadata"]["scenario"], scenario)

    def test_generation_is_deterministic_and_hits_requested_size(self):
        first, first_counts = generate(
            [RECORD], sample_count=50, seed=9, canvas=840, scenario_weights=(1, 1, 1, 1, 1, 1)
        )
        second, second_counts = generate(
            [RECORD], sample_count=50, seed=9, canvas=840, scenario_weights=(1, 1, 1, 1, 1, 1)
        )
        self.assertEqual(first, second)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(len(first), 50)
        self.assertEqual(sum(first_counts.values()), 50)


if __name__ == "__main__":
    unittest.main()
