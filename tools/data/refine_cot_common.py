"""Shared parsing helpers for Refine CoT request construction and merging."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ASSISTANT_RE = re.compile(
    r"^\s*<think>(?P<cot>.*?)</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
IMAGE_TAG_RE = re.compile(r"<img>(?P<path>.*?)</img>", re.DOTALL | re.IGNORECASE)


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise TypeError("top-level JSON value must be a list")
        return records, "json"
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records, "jsonl"


def write_records(path: Path, records: list[dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sample_fields(record: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return style, image path, user content and assistant content."""
    messages = record.get("messages")
    if isinstance(messages, list):
        user = next((item for item in messages if item.get("role") == "user"), None)
        assistant = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
        images = record.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError("messages-format sample must contain exactly one string image path")
        if not user or not assistant:
            raise ValueError("messages-format sample requires user and assistant messages")
        return "messages", images[0], str(user.get("content", "")), str(assistant.get("content", ""))

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        user = next((item for item in conversations if item.get("from") in {"user", "human"}), None)
        assistant = next(
            (item for item in reversed(conversations) if item.get("from") in {"assistant", "gpt"}), None
        )
        if not user or not assistant:
            raise ValueError("conversations-format sample requires user and assistant turns")
        user_content = str(user.get("value", ""))
        image_match = IMAGE_TAG_RE.search(user_content)
        if image_match is None:
            raise ValueError("legacy user turn is missing <img>path</img>")
        return "conversations", image_match.group("path").strip(), user_content, str(assistant.get("value", ""))

    raise ValueError("sample has neither messages nor conversations format")


def parse_assistant(content: str) -> tuple[str, str, dict[str, Any]]:
    match = ASSISTANT_RE.fullmatch(content)
    if match is None:
        raise ValueError("assistant content must strictly contain <think>...</think><answer>...</answer>")
    answer_text = match.group("answer").strip()
    answer_obj = json.loads(answer_text)
    if not isinstance(answer_obj, dict):
        raise TypeError("refine answer must be a JSON object")
    if set(answer_obj) != {"per_bbox", "new_items"}:
        raise ValueError("refine answer must contain exactly per_bbox and new_items")
    if not isinstance(answer_obj["per_bbox"], list) or not isinstance(answer_obj["new_items"], list):
        raise TypeError("per_bbox and new_items must be lists")
    return match.group("cot").strip(), answer_text, answer_obj


def request_id_for(record: dict[str, Any]) -> str:
    _, image, user_content, assistant_content = sample_fields(record)
    _, answer_text, _ = parse_assistant(assistant_content)
    source_id = str((record.get("metadata") or {}).get("source_id", record.get("source", "")))
    payload = json.dumps(
        {"source_id": source_id, "image": image, "user": user_content, "answer": answer_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replace_cot(record: dict[str, Any], cot: str, request_id: str, api_model: str | None) -> dict[str, Any]:
    style, _, _, assistant_content = sample_fields(record)
    _, answer_text, _ = parse_assistant(assistant_content)
    new_content = f"<think>\n{cot.strip()}\n</think>\n<answer>\n{answer_text}\n</answer>"
    output = json.loads(json.dumps(record, ensure_ascii=False))

    if style == "messages":
        assistant = next(item for item in reversed(output["messages"]) if item.get("role") == "assistant")
        assistant["content"] = new_content
        metadata = output.setdefault("metadata", {})
    else:
        assistant = next(
            item for item in reversed(output["conversations"]) if item.get("from") in {"assistant", "gpt"}
        )
        assistant["value"] = new_content
        metadata = output.setdefault("cot_metadata", {})

    metadata["cot_source"] = "company_vlm_api"
    metadata["cot_request_id"] = request_id
    if api_model:
        metadata["cot_api_model"] = api_model
    return output
