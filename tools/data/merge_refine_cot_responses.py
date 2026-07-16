#!/usr/bin/env python3
"""Merge company-VLM CoT responses while preserving verified Refine answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tools.data.refine_cot_common import load_records, replace_cot, request_id_for, write_records


FORBIDDEN_RE = re.compile(r"</?(?:think|answer)>|VERIFIED_REFINE_ACTION", re.IGNORECASE)


def response_cot(record: dict[str, Any]) -> str:
    value: Any = record.get("cot")
    if value is None and isinstance(record.get("output"), dict):
        value = record["output"].get("cot")
    if not isinstance(value, str):
        raise TypeError("response must contain a string cot field")
    return value.strip()


def validate_cot(cot: str, min_chars: int, max_chars: int) -> None:
    if len(cot) < min_chars:
        raise ValueError(f"CoT is too short: {len(cot)} < {min_chars}")
    if len(cot) > max_chars:
        raise ValueError(f"CoT is too long: {len(cot)} > {max_chars}")
    if FORBIDDEN_RE.search(cot):
        raise ValueError("CoT contains forbidden answer/tags")


def load_responses(path: Path, min_chars: int, max_chars: int) -> dict[str, tuple[str, str | None]]:
    responses, _ = load_records(path)
    indexed: dict[str, tuple[str, str | None]] = {}
    for line_no, response in enumerate(responses, 1):
        request_id = response.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"response {line_no}: missing request_id")
        if request_id in indexed:
            raise ValueError(f"response {line_no}: duplicate request_id {request_id}")
        cot = response_cot(response)
        validate_cot(cot, min_chars, max_chars)
        api_model = response.get("model", response.get("api_model"))
        indexed[request_id] = (cot, str(api_model) if api_model else None)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True, help="The exact draft used to export requests")
    parser.add_argument("--responses", type=Path, required=True, help="JSONL/JSON responses with request_id and cot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-format", choices=("auto", "jsonl", "json"), default="auto")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=1600)
    args = parser.parse_args()

    drafts, draft_format = load_records(args.draft)
    responses = load_responses(args.responses, args.min_chars, args.max_chars)
    merged: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for index, draft in enumerate(drafts, 1):
        request_id = request_id_for(draft)
        if request_id not in responses:
            raise ValueError(f"draft {index}: missing response for {request_id}")
        cot, api_model = responses[request_id]
        merged.append(replace_cot(draft, cot, request_id, api_model))
        consumed.add(request_id)
    unused = set(responses) - consumed
    if unused:
        raise ValueError(f"responses contain {len(unused)} unused request ids")

    output_format = draft_format if args.output_format == "auto" else args.output_format
    write_records(args.output, merged, output_format)
    print(json.dumps({"merged": len(merged), "format": output_format, "output": str(args.output)}))


if __name__ == "__main__":
    main()
