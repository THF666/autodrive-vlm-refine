#!/usr/bin/env python3
"""Export provider-neutral company-VLM requests from verified Refine SFT drafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.data.refine_cot_common import load_records, parse_assistant, request_id_for, sample_fields


SYSTEM_PROMPT = """You are constructing supervised visual reasoning data for a 2D detection Refine task.
Inspect the image, target query, previous predictions, and the verified refine action.
Explain visually why each candidate should be corrected, kept, or deleted and why any target is missing.
The refine action is an immutable oracle produced from ground-truth boxes: do not change, recompute, or repeat its numbers.
Return strict JSON with keys request_id and cot only. The cot must be concise, image-grounded, and contain no <think>, <answer>, or JSON answer."""


def action_summary(answer_obj: dict[str, Any]) -> dict[str, Any]:
    corrected: list[str] = []
    deleted: list[str] = []
    for item in answer_obj["per_bbox"]:
        action = item.get("action") if isinstance(item, dict) else None
        if action == "delete":
            deleted.append(str(item.get("id", "")))
        elif isinstance(action, dict):
            corrected.append(str(item.get("id", "")))
    return {"correct_or_adjust": corrected, "delete": deleted, "add_count": len(answer_obj["new_items"])}


def make_request(record: dict[str, Any]) -> dict[str, Any]:
    _, image, user_content, assistant_content = sample_fields(record)
    _, answer_text, answer_obj = parse_assistant(assistant_content)
    request_id = request_id_for(record)
    user_prompt = (
        f"{user_content}\n\n"
        "The following refine action has already been computed and replay-validated against ground truth. "
        "Use it only to understand which visual correction is required; do not alter it.\n"
        f"VERIFIED_REFINE_ACTION={answer_text}\n"
        f"ACTION_SUMMARY={json.dumps(action_summary(answer_obj), ensure_ascii=False, separators=(',', ':'))}\n"
        f"Return: {{\"request_id\":\"{request_id}\",\"cot\":\"one concise image-grounded rationale\"}}"
    )
    return {
        "request_id": request_id,
        "image": image,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "oracle_answer": answer_obj,
        "action_summary": action_summary(answer_obj),
        "source_metadata": record.get("metadata", record.get("source", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Verified draft SFT JSONL or legacy JSON list")
    parser.add_argument("--output", type=Path, required=True, help="Provider-neutral VLM request JSONL")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records, _ = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    requests = [make_request(record) for record in records]
    request_ids = [item["request_id"] for item in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate CoT request ids detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"requests": len(requests), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
