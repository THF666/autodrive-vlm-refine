import json
import tempfile
import unittest
from pathlib import Path

from tools.data.merge_refine_cot_responses import load_responses
from tools.data.prepare_refine_cot_requests import make_request
from tools.data.refine_cot_common import replace_cot, request_id_for, sample_fields


SAMPLE = {
    "messages": [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": '<image>Locate "vehicle". Previous predictions: [{"id":"B1"}]',
        },
        {
            "role": "assistant",
            "content": (
                "<think>template</think><answer>"
                '{"per_bbox":[{"id":"B1","action":{"bbox":[1,2,3,4],"point":[1,2]}}],'
                '"new_items":[]}</answer>'
            ),
        },
    ],
    "images": ["/private/frame.jpg"],
    "metadata": {"source_id": "sample-1", "scenario": "jitter"},
}


class RefineCotPipelineTests(unittest.TestCase):
    def test_request_contains_image_and_immutable_oracle(self):
        request = make_request(SAMPLE)
        self.assertEqual(request["image"], "/private/frame.jpg")
        self.assertEqual(request["request_id"], request_id_for(SAMPLE))
        self.assertIn("VERIFIED_REFINE_ACTION", request["messages"][1]["content"])
        self.assertEqual(request["action_summary"]["correct_or_adjust"], ["B1"])

    def test_merge_replaces_only_cot_and_keeps_answer(self):
        request_id = request_id_for(SAMPLE)
        cot = "The candidate covers the vehicle but its boundaries need a small visual adjustment."
        merged = replace_cot(SAMPLE, cot, request_id, "company-vlm")
        _, _, _, assistant = sample_fields(merged)
        self.assertIn(cot, assistant)
        self.assertIn('"bbox":[1,2,3,4]', assistant)
        self.assertEqual(merged["metadata"]["cot_source"], "company_vlm_api")

    def test_response_loader_rejects_answer_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(
                json.dumps({"request_id": "x", "cot": "<answer>do not replace the oracle answer</answer>"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_responses(path, min_chars=1, max_chars=1600)


if __name__ == "__main__":
    unittest.main()
