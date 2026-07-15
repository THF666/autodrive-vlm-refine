# -*- coding: utf-8 -*-
"""
Simple Propose-Refine reward.

Reward = R_format + R_final + R_improve, with no negative penalty.
R_final follows the VisionReasoner-style bbox IoU / bbox L1 / point L1
signals after applying refine actions to the proposal state.
"""

import re
import json
import math
import os
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment

# ============================================================
# 0. Config
# ============================================================
DEBUG_PRINT = os.getenv("PROPOSE_REFINE_REWARD_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

# ---------------- Strictness Toggles ----------------
STRICT_ASSISTANT_SHAPE = True     # Must match <think>...</think><answer>...</answer> exactly
REQUIRE_NEW_ITEM_POINT = True     # New items MUST have 'point_2d'
REQUIRE_REFINE_ACTION_POINT = True  # Refine per_bbox action dict MUST have 'point'
REQUIRE_REFINE_KEYS = True        # Refine MUST contain both keys: per_bbox and new_items
STRICT_PER_BBOX_ACTION = True     # Disallow legacy action encodings (e.g., list deltas)

# ---------------- Reward Weights ----------------
DEFAULT_WEIGHTS = {
    "format": 1.0,
    "final": 1.0,
    "improve": 1.0,
}

BBOX_IOU_THRESHOLD = 0.5
BBOX_EDGE_L1_THRESHOLD = 10.0
POINT_L1_THRESHOLD = 30.0
IMPROVE_EPS = 1e-6
IMPROVE_RATIO_LEVEL2_THRESHOLD = 0.25
IMPROVE_RATIO_LEVEL3_THRESHOLD = 0.50
CANVAS_SIZE = int(os.getenv("PROPOSE_REFINE_CANVAS_SIZE", "840"))

# ============================================================
# 1. Entry Point
# ============================================================
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> Dict[str, float]:
    extra = extra_info or {}
    w = {**DEFAULT_WEIGHTS, **(extra.get("reward_weights") or {})}
    # Multi-turn rollout masks the proposal turn out of the PPO loss, but the
    # reward still needs both assistant replies to replay proposal -> refine.
    # tool_agent_loop stores those raw replies in assistant_responses.
    solution_for_reward = _solution_from_assistant_responses(extra, solution_str)

    gt_data = _parse_gt_safe(ground_truth)
    last_json_obj = _get_last_turn_json_strict(solution_for_reward)
    last_response = _last_assistant_response(solution_for_reward)

    # The final trainable turn is expected to be a refine action. If the last
    # answer is malformed, return a full zero reward instead of partial tag
    # credit; this keeps the RL signal focused on executable refine JSON.
    format_score = 1.0 if _is_executable_refine_answer(last_json_obj) else 0.0
    if format_score <= 0.0:
        if DEBUG_PRINT:
            print("[Reward] format=0.0000 final=0.0000 improve=0.0000 total=0.0000")
            print("[Reward][response]")
            print(last_response)
        return _reward_result(
            score=0.0,
            format_score=0.0,
            final_score=0.0,
            improve_score=0.0,
            proposal_iou=0.0,
            refine_iou=0.0,
            proposal_bbox_l1_hit=0.0,
            refine_bbox_l1_hit=0.0,
            proposal_point_hit=0.0,
            refine_point_hit=0.0,
            refine_bbox_iou_hit=0.0,
            steps=0.0,
        )

    states = _replay_states(solution_for_reward, extra_info)
    if not states:
        total = float(w["format"] * format_score)
        if DEBUG_PRINT:
            print("[Reward] format=1.0000 final=0.0000 improve=0.0000 total=1.0000 states=0")
            print("[Reward][response]")
            print(last_response)
        return _reward_result(
            score=total,
            format_score=format_score,
            final_score=0.0,
            improve_score=0.0,
            proposal_iou=0.0,
            refine_iou=0.0,
            proposal_bbox_l1_hit=0.0,
            refine_bbox_l1_hit=0.0,
            proposal_point_hit=0.0,
            refine_point_hit=0.0,
            refine_bbox_iou_hit=0.0,
            steps=0.0,
        )

    final_metrics = _localization_metrics(states[-1], gt_data)
    final_score = _localization_score(final_metrics)

    # Improvement is computed against the state immediately before the final
    # refine state. In the normal two-turn RL path this is the model proposal;
    # in one-turn-refine mode it is extra_info["proposal"].
    improve_score = 0.0
    improve_gain_ratio = 0.0
    improve_iou_gain_ratio = 0.0
    improve_l1_gain_ratio = 0.0
    prev_metrics = None
    if len(states) >= 2:
        prev_metrics = _localization_metrics(states[-2], gt_data)
        improve_gain_ratio, improve_iou_gain_ratio, improve_l1_gain_ratio = _relative_improve_ratio(
            prev_metrics, final_metrics
        )
        if improve_gain_ratio > IMPROVE_EPS:
            if improve_gain_ratio >= IMPROVE_RATIO_LEVEL3_THRESHOLD:
                improve_score = 3.0
            elif improve_gain_ratio >= IMPROVE_RATIO_LEVEL2_THRESHOLD:
                improve_score = 2.0
            else:
                improve_score = 1.0

    total = w["format"] * format_score + w["final"] * final_score + w["improve"] * improve_score

    proposal_iou = prev_metrics["mean_iou"] if prev_metrics is not None else 0.0
    proposal_bbox_l1 = prev_metrics["mean_bbox_l1"] if prev_metrics is not None else 0.0
    proposal_bbox_l1_hit = prev_metrics["bbox_l1_hit"] if prev_metrics is not None else 0.0
    proposal_point_hit = prev_metrics["point_hit"] if prev_metrics is not None else 0.0

    if DEBUG_PRINT:
        print(
            "[Reward] "
            f"format={format_score:.4f} final={final_score:.4f} improve={improve_score:.4f} "
            f"total={total:.4f} "
            f"proposal_iou={proposal_iou} refine_iou={final_metrics['mean_iou']:.4f} "
            f"improve_gain_ratio={improve_gain_ratio:.4f} "
            f"improve_iou_gain_ratio={improve_iou_gain_ratio:.4f} "
            f"improve_l1_gain_ratio={improve_l1_gain_ratio:.4f} "
            f"proposal_bbox_l1={proposal_bbox_l1:.4f} refine_bbox_l1={final_metrics['mean_bbox_l1']:.4f} "
            f"proposal_bbox_l1_hit={proposal_bbox_l1_hit} refine_bbox_l1_hit={final_metrics['bbox_l1_hit']:.4f} "
            f"proposal_point_hit={proposal_point_hit} refine_point_hit={final_metrics['point_hit']:.4f} "
            f"refine_bbox_iou_hit={final_metrics['bbox_iou_hit']:.4f} "
            f"steps={len(states)}"
        )
        print("[Reward][response]")
        print(last_response)

    return _reward_result(
        score=float(total),
        format_score=float(format_score),
        final_score=float(final_score),
        improve_score=float(improve_score),
        proposal_iou=float(proposal_iou),
        refine_iou=float(final_metrics["mean_iou"]),
        proposal_bbox_l1_hit=float(proposal_bbox_l1_hit),
        refine_bbox_l1_hit=float(final_metrics["bbox_l1_hit"]),
        proposal_point_hit=float(proposal_point_hit),
        refine_point_hit=float(final_metrics["point_hit"]),
        refine_bbox_iou_hit=float(final_metrics["bbox_iou_hit"]),
        steps=float(len(states)),
        improve_gain_ratio=float(improve_gain_ratio),
        improve_iou_gain_ratio=float(improve_iou_gain_ratio),
        improve_l1_gain_ratio=float(improve_l1_gain_ratio),
        proposal_bbox_l1=float(proposal_bbox_l1),
        refine_bbox_l1=float(final_metrics["mean_bbox_l1"]),
        gt_count=float(len(gt_data)),
        proposal_count=float(len(states[-2]) if len(states) >= 2 else 0),
        refine_count=float(len(states[-1])),
    )


def _reward_result(
    score: float,
    format_score: float,
    final_score: float,
    improve_score: float,
    proposal_iou: float,
    refine_iou: float,
    proposal_bbox_l1_hit: float,
    refine_bbox_l1_hit: float,
    proposal_point_hit: float,
    refine_point_hit: float,
    refine_bbox_iou_hit: float,
    steps: float,
    improve_gain_ratio: float = 0.0,
    improve_iou_gain_ratio: float = 0.0,
    improve_l1_gain_ratio: float = 0.0,
    proposal_bbox_l1: float = 0.0,
    refine_bbox_l1: float = 0.0,
    gt_count: float = 0.0,
    proposal_count: float = 0.0,
    refine_count: float = 0.0,
) -> Dict[str, float]:
    return {
        "score": float(score),
        "format": float(format_score),
        "final": float(final_score),
        "improve": float(improve_score),
        "proposal_iou": float(proposal_iou),
        "refine_iou": float(refine_iou),
        "proposal_bbox_l1_hit": float(proposal_bbox_l1_hit),
        "refine_bbox_l1_hit": float(refine_bbox_l1_hit),
        "proposal_point_hit": float(proposal_point_hit),
        "refine_point_hit": float(refine_point_hit),
        "refine_bbox_iou_hit": float(refine_bbox_iou_hit),
        "steps": float(steps),
        "improve_gain_ratio": float(improve_gain_ratio),
        "improve_iou_gain_ratio": float(improve_iou_gain_ratio),
        "improve_l1_gain_ratio": float(improve_l1_gain_ratio),
        "proposal_bbox_l1": float(proposal_bbox_l1),
        "refine_bbox_l1": float(refine_bbox_l1),
        "gt_count": float(gt_count),
        "proposal_count": float(proposal_count),
        "refine_count": float(refine_count),
    }


# ============================================================
# 2. Parsing Utilities
# ============================================================
def _solution_from_assistant_responses(extra_info: dict, fallback_solution: str) -> str:
    responses = (extra_info or {}).get("assistant_responses")
    if responses is None:
        tool_extra_fields = (extra_info or {}).get("tool_extra_fields")
        if isinstance(tool_extra_fields, np.ndarray):
            tool_extra_fields = tool_extra_fields.tolist()
        if isinstance(tool_extra_fields, list) and len(tool_extra_fields) == 1:
            tool_extra_fields = tool_extra_fields[0]
        if isinstance(tool_extra_fields, dict):
            responses = tool_extra_fields.get("assistant_responses")
    if responses is None:
        return fallback_solution or ""
    if isinstance(responses, np.ndarray):
        responses = responses.tolist()
    if isinstance(responses, tuple):
        responses = list(responses)
    if not isinstance(responses, list):
        return fallback_solution or ""

    clean = [str(r or "").strip() for r in responses if str(r or "").strip()]
    if not clean:
        return fallback_solution or ""
    return "\n".join(f"assistant\n{text}" for text in clean)

# def _split_turns_robust(log_str: str) -> List[Tuple[str, str]]:
#     """Robust splitter for User/Assistant turns."""
#     if "<|im_start|>" in log_str:
#         pattern = r"<\|im_start\|>\s*(user|assistant|system)\s*"
#     else:
#         pattern = r"(User:|Assistant:|System:)"

#     parts = re.split(pattern, log_str, flags=re.IGNORECASE)
#     turns = []
#     current_role = None

#     for p in parts:
#         match = re.match(r"^(user|assistant|system)$", p.strip(":").lower(), re.IGNORECASE)
#         if match:
#             current_role = match.group(1).lower()
#         elif current_role:
#             turns.append((current_role, p.strip()))
#             current_role = None

#     if not turns and "<answer>" in log_str:
#         turns.append(("assistant", log_str))

#     return turns
def _split_turns_robust(log_str: str) -> List[Tuple[str, str]]:
    """
    Robust splitter for User/Assistant turns.

    Supported formats:
    1) <|im_start|> user/assistant/system ...
    2) Lines starting with "User:" / "Assistant:" / "System:"
    3) A line that is exactly "user" / "assistant" / "system"
       (also allows a single leading prefix like "(TaskRunner pid=...) user")
    """
    # -----------------------------
    # Case 1: Qwen-style <|im_start|>
    # -----------------------------
    if "<|im_start|>" in log_str:
        pattern = r"<\|im_start\|>\s*(user|assistant|system)\s*"
        parts = re.split(pattern, log_str, flags=re.IGNORECASE)
        turns: List[Tuple[str, str]] = []
        current_role = None
        for p in parts:
            m = re.match(r"^(user|assistant|system)$", p.strip().lower())
            if m:
                current_role = m.group(1).lower()
                continue
            if current_role is not None:
                turns.append((current_role, p.strip()))
                current_role = None
        if not turns and "<answer>" in log_str:
            turns.append(("assistant", log_str.strip()))
        return turns

    # ------------------------------------------------
    # Case 2: "User:" / "Assistant:" / "System:" lines
    # ------------------------------------------------
    if re.search(r"(?mi)^\s*(User:|Assistant:|System:)\s*", log_str):
        pattern = r"(?mi)^\s*(User:|Assistant:|System:)\s*"
        parts = re.split(pattern, log_str)
        turns: List[Tuple[str, str]] = []
        current_role = None
        for p in parts:
            tag = p.strip().lower()
            if tag in ("user:", "assistant:", "system:"):
                current_role = tag[:-1]  # remove ':'
                continue
            if current_role is not None:
                turns.append((current_role, p.strip()))
                current_role = None
        if not turns and "<answer>" in log_str:
            turns.append(("assistant", log_str.strip()))
        return turns

    # ---------------------------------------------------------
    # Case 3: role marker is a whole line: "user" / "assistant"
    # Also tolerates a leading "(...)" prefix: "(TaskRunner ...) user"
    # ---------------------------------------------------------
    role_line_pat = re.compile(
        r"(?i)^\s*(?:\([^)\n]*\)\s*)*(user|assistant|system)\s*$"
    )

    lines = log_str.splitlines()
    turns: List[Tuple[str, str]] = []

    current_role: Optional[str] = None
    buf: List[str] = []
    prebuf: List[str] = []

    def _flush(role: Optional[str], buffer: List[str]):
        if role is None:
            return
        content = "\n".join(buffer).strip()
        if content:
            turns.append((role, content))

    for line in lines:
        m = role_line_pat.match(line)
        if m:
            new_role = m.group(1).lower()

            # If we have prebuf before seeing the first explicit role marker,
            # treat it as assistant if it looks like an assistant response.
            if current_role is None and prebuf:
                pre_text = "\n".join(prebuf).strip()
                if "<think>" in pre_text or "<answer>" in pre_text:
                    turns.append(("assistant", pre_text))
                prebuf = []

            # Flush previous role buffer
            if current_role is not None:
                _flush(current_role, buf)
                buf = []

            current_role = new_role
            continue

        # Normal line
        if current_role is None:
            prebuf.append(line)
        else:
            buf.append(line)

    # Flush tail
    if current_role is not None:
        _flush(current_role, buf)
    else:
        # No explicit markers at all, fallback
        if "<answer>" in log_str:
            turns.append(("assistant", log_str.strip()))

    return turns

def _check_tags_robust(text: str) -> bool:
    turns = _split_turns_robust(text)
    if not turns:
        return False
    _, last_content = turns[-1]

    has_tags = ("<think>" in last_content and "</think>" in last_content and
                "<answer>" in last_content and "</answer>" in last_content)

    if not has_tags:
        return False

    if STRICT_ASSISTANT_SHAPE:
        if not re.search(r"</answer>\s*$", last_content, re.DOTALL):
            return False
    return True

def _get_last_turn_json_strict(text: str) -> Optional[Any]:
    turns = _split_turns_robust(text)
    if not turns:
        return None
    _, last_content = turns[-1]

    _, ans_text = _extract_think_answer(last_content)
    if not ans_text:
        return None

    json_obj = _parse_json_strict(ans_text)
    if json_obj is None:
        return None

    # --- Schema Validation ---
    # PROPOSE: list of {"bbox_2d":[4], "point_2d":[2]}
    if isinstance(json_obj, list):
        for it in json_obj:
            if not isinstance(it, dict):
                return None
            if not _is_valid_box4(it.get("bbox_2d")):
                return None
            if REQUIRE_NEW_ITEM_POINT and not _is_valid_point2(it.get("point_2d")):
                return None
        return json_obj

    # REFINE: dict with REQUIRED keys per_bbox and new_items, strict per item schema
    if isinstance(json_obj, dict):
        if REQUIRE_REFINE_KEYS:
            if "per_bbox" not in json_obj or "new_items" not in json_obj:
                return None
        else:
            if "per_bbox" not in json_obj and "new_items" not in json_obj:
                return None

        # Validate per_bbox
        if "per_bbox" in json_obj:
            if not isinstance(json_obj["per_bbox"], list):
                return None
            for item in json_obj["per_bbox"]:
                if not isinstance(item, dict):
                    return None
                if "id" not in item or "action" not in item:
                    return None
                if not isinstance(item["id"], str) or len(item["id"]) == 0:
                    return None

                act = item["action"]
                # Allowed: "delete" OR {"bbox":[4], "point":[2]}
                if isinstance(act, str):
                    if act != "delete":
                        return None
                elif isinstance(act, dict):
                    if "bbox" not in act or "point" not in act:
                        return None
                    if not _is_valid_box4(act.get("bbox")):
                        return None
                    if REQUIRE_REFINE_ACTION_POINT and not _is_valid_point2(act.get("point")):
                        return None
                else:
                    return None

        # Validate new_items
        if "new_items" in json_obj:
            if not isinstance(json_obj["new_items"], list):
                return None
            for item in json_obj["new_items"]:
                if not isinstance(item, dict):
                    return None
                if not _is_valid_box4(item.get("bbox_2d")):
                    return None
                if REQUIRE_NEW_ITEM_POINT and not _is_valid_point2(item.get("point_2d")):
                    return None

        return json_obj

    return None

def _last_assistant_response(solution_str: str) -> str:
    turns = _split_turns_robust(solution_str)
    for role, content in reversed(turns):
        if role == "assistant":
            return content
    return solution_str or ""

def _parse_gt_safe(ground_truth: Any) -> List[Dict]:
    if isinstance(ground_truth, list):
        return _normalize_gt_items(ground_truth)
    if isinstance(ground_truth, dict):
        if "boxes" in ground_truth:
            return _normalize_gt_items(ground_truth["boxes"])
        return _normalize_gt_items([ground_truth])
    if not isinstance(ground_truth, str):
        raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)!r}")

    text = ground_truth.strip()
    if not text:
        raise ValueError("Empty ground_truth.")
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict) and "boxes" in data:
            data = data["boxes"]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"JSON ground_truth must be a list or dict, got: {type(data)!r}")
        return _normalize_gt_items(data)
    if "<box>" in text.lower():
        return _parse_visionreasoner_gt(text)
    raise ValueError(f"Unsupported ground_truth format: {text[:200]!r}")

def _normalize_gt_items(items: List[Any]) -> List[Dict]:
    out: List[Dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"GT item {i} must be dict, got {type(item)!r}")
        bbox = item.get("bbox_2d", item.get("bbox"))
        if not _is_valid_box4(bbox):
            raise ValueError(f"GT item {i} missing valid bbox_2d/bbox: {item!r}")
        point = item.get("point_2d", item.get("point"))
        box = [float(v) for v in bbox]
        if _is_valid_point2(point):
            pt = [float(point[0]), float(point[1])]
        else:
            pt = _box_center(box)
        out.append({"bbox_2d": box, "point_2d": pt})
    if not out:
        raise ValueError("No valid GT boxes parsed from ground_truth.")
    return out

def _parse_visionreasoner_gt(text: str) -> List[Dict]:
    box_blocks = re.findall(r"<box>\s*(.*?)\s*</box>", text, flags=re.IGNORECASE | re.DOTALL)
    point_blocks = re.findall(r"<points?>\s*(.*?)\s*</points?>", text, flags=re.IGNORECASE | re.DOTALL)
    if not box_blocks:
        raise ValueError(f"VisionReasoner ground_truth has no <box> block: {text[:200]!r}")

    boxes: List[List[float]] = []
    for block in box_blocks:
        pairs = _parse_coord_pairs(block)
        if len(pairs) != 2:
            raise ValueError(f"Each <box> block must contain exactly two coordinate pairs, got {block!r}")
        (x1, y1), (x2, y2) = pairs
        boxes.append([x1, y1, x2, y2])

    points: List[List[float]] = []
    for block in point_blocks:
        points.extend([[x, y] for x, y in _parse_coord_pairs(block)])

    out: List[Dict] = []
    for i, box in enumerate(boxes):
        point = points[i] if i < len(points) else _box_center(box)
        out.append({"bbox_2d": box, "point_2d": point})
    return out

def _parse_coord_pairs(text: str) -> List[Tuple[float, float]]:
    pairs = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", text)
    if not pairs:
        raise ValueError(f"No coordinate pairs found in {text!r}")
    return [(float(x), float(y)) for x, y in pairs]

def _is_executable_refine_answer(obj: Any) -> bool:
    return _validate_refine_obj_strict(obj) is not None

def _extract_think_answer(text: str) -> Tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return think, answer

def _parse_json_strict(text: str) -> Any:
    if not text:
        return None
    text = re.sub(r"^```json", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except:
        return None


# ============================================================
# 3. State Replay (Strict Refine Schema)
# ============================================================
def _replay_states(solution_str: str, extra_info: dict) -> List[List[Dict[str, Any]]]:
    turns = _split_turns_robust(solution_str)
    states: List[List[Dict[str, Any]]] = []
    # For cold-start-like one-turn refine, the proposal is supplied by the data
    # row rather than generated by the first assistant turn.
    current_items: List[Dict[str, Any]] = _proposal_state_from_extra(extra_info)
    if current_items:
        states.append(json.loads(json.dumps(current_items)))

    for role, content in turns:
        if role != "assistant":
            continue

        _, ans_text = _extract_think_answer(content)
        json_obj = _parse_json_strict(ans_text)
        if not json_obj:
            continue

        # --- PROPOSE PHASE ---
        if isinstance(json_obj, list):
            valid_items = []
            for it in json_obj:
                if isinstance(it, dict) and _is_valid_box4(it.get("bbox_2d")):
                    if REQUIRE_NEW_ITEM_POINT and not _is_valid_point2(it.get("point_2d")):
                        continue
                    it = dict(it)
                    it["bbox_2d"] = _clamp_box_abs(it["bbox_2d"])
                    it["point_2d"] = _clamp_point_abs(it.get("point_2d", [0, 0]))
                    valid_items.append(it)
            current_items = valid_items
            states.append(json.loads(json.dumps(current_items)))
            continue

        # --- REFINE PHASE ---
        if isinstance(json_obj, dict):
            # Hard gate: must pass strict schema for refine to be applied
            if _validate_refine_obj_strict(json_obj) is None:
                continue
            current_items = _apply_refine_action(current_items, json_obj)
            states.append(json.loads(json.dumps(current_items)))

    return states


def _proposal_state_from_extra(extra_info: dict) -> List[Dict[str, Any]]:
    proposal = (extra_info or {}).get("proposal")
    if proposal is None:
        return []
    if isinstance(proposal, np.ndarray):
        proposal = proposal.tolist()
    if isinstance(proposal, str):
        proposal = json.loads(proposal)
    if not isinstance(proposal, list):
        raise TypeError(f"extra_info['proposal'] must be a list or JSON list string, got {type(proposal)!r}")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(proposal):
        if not isinstance(item, dict):
            raise TypeError(f"proposal item {i} must be dict, got {type(item)!r}")
        box = item.get("bbox_2d", item.get("bbox"))
        if not _is_valid_box4(box):
            raise ValueError(f"proposal item {i} missing valid bbox_2d/bbox: {item!r}")
        point = item.get("point_2d", item.get("point"))
        if _is_valid_point2(point):
            point_out = [int(round(float(point[0]))), int(round(float(point[1])))]
        else:
            point_out = _box_center([float(v) for v in box])
        out.append({
            "id": str(item.get("id") or f"B{i + 1}"),
            "bbox_2d": _clamp_box_abs(box),
            "point_2d": _clamp_point_abs(point_out),
        })
    if not out:
        raise ValueError("extra_info['proposal'] is empty after normalization.")
    return out

def _validate_refine_obj_strict(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if REQUIRE_REFINE_KEYS:
        if "per_bbox" not in obj or "new_items" not in obj:
            return None

    # per_bbox
    per_bbox = obj.get("per_bbox", [])
    if not isinstance(per_bbox, list):
        return None
    for op in per_bbox:
        if not isinstance(op, dict):
            return None
        if "id" not in op or "action" not in op:
            return None
        if not isinstance(op["id"], str) or len(op["id"]) == 0:
            return None
        act = op["action"]
        if isinstance(act, str):
            if act != "delete":
                return None
        elif isinstance(act, dict):
            if "bbox" not in act or "point" not in act:
                return None
            if not _is_valid_box4(act.get("bbox")):
                return None
            if REQUIRE_REFINE_ACTION_POINT and not _is_valid_point2(act.get("point")):
                return None
        else:
            return None

    # new_items
    new_items = obj.get("new_items", [])
    if not isinstance(new_items, list):
        return None
    for it in new_items:
        if not isinstance(it, dict):
            return None
        if not _is_valid_box4(it.get("bbox_2d")):
            return None
        if REQUIRE_NEW_ITEM_POINT and not _is_valid_point2(it.get("point_2d")):
            return None

    return obj

def _apply_refine_action(prev_items: List[Dict], action_obj: Dict) -> List[Dict]:
    """
    Strict refine format:
      per_bbox: [{"id":"B1","action":"delete"} OR {"id":"B1","action":{"bbox":[4], "point":[2]}}]
      new_items: [{"bbox_2d":[4], "point_2d":[2]}]
    """
    id_map: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(prev_items):
        bid = str(item.get("id") or f"B{i + 1}")
        copied = dict(item)
        copied["id"] = bid
        id_map[bid] = copied

    per_bbox = action_obj.get("per_bbox", [])
    if isinstance(per_bbox, list):
        for op in per_bbox:
            if not isinstance(op, dict):
                continue
            bid = op.get("id")
            act = op.get("action")
            if bid not in id_map:
                continue

            if act == "delete":
                id_map[bid]["_deleted"] = True
                continue

            # Strict: act must be dict with bbox and point
            if STRICT_PER_BBOX_ACTION:
                if not isinstance(act, dict):
                    continue
                deltas = act.get("bbox")
                p_deltas = act.get("point")
                if not _is_valid_box4(deltas):
                    continue
                if REQUIRE_REFINE_ACTION_POINT and not _is_valid_point2(p_deltas):
                    continue
            else:
                # Legacy fallback (kept but normally disabled)
                deltas = [0, 0, 0, 0]
                if isinstance(act, list) and len(act) == 4:
                    deltas = act
                elif isinstance(act, dict) and "bbox" in act:
                    deltas = act["bbox"]
                p_deltas = [0, 0]
                if isinstance(act, dict) and "point" in act:
                    p_deltas = act["point"]

            # Apply BBox delta
            old_b = id_map[bid].get("bbox_2d", [0, 0, 0, 0])
            new_b = [int(old_b[i]) + int(round(float(deltas[i]))) for i in range(4)]
            id_map[bid]["bbox_2d"] = _clamp_box_abs(new_b)

            # Apply Point delta (no inheritance if missing, but we already validated it exists)
            if "point_2d" in id_map[bid] and _is_valid_point2(p_deltas):
                old_p = id_map[bid]["point_2d"]
                new_p = [
                    int(old_p[0]) + int(round(float(p_deltas[0]))),
                    int(old_p[1]) + int(round(float(p_deltas[1]))),
                ]
                id_map[bid]["point_2d"] = _clamp_point_abs(new_p)

    new_list = []
    def _sort_key(bid: str):
        m = re.fullmatch(r"B(\d+)", str(bid))
        return (0, int(m.group(1))) if m else (1, str(bid))

    for k in sorted(id_map.keys(), key=_sort_key):
        if not id_map[k].get("_deleted"):
            # Clean internal marker if present
            if "_deleted" in id_map[k]:
                id_map[k].pop("_deleted", None)
            new_list.append(id_map[k])

    # Process Adds
    adds = action_obj.get("new_items", [])
    if isinstance(adds, list):
        for it in adds:
            if isinstance(it, dict) and _is_valid_box4(it.get("bbox_2d")):
                if REQUIRE_NEW_ITEM_POINT and not _is_valid_point2(it.get("point_2d")):
                    continue
                it2 = dict(it)
                it2["bbox_2d"] = _clamp_box_abs(it2["bbox_2d"])
                it2["point_2d"] = _clamp_point_abs(it2.get("point_2d", [0, 0]))
                it2["id"] = str(it2.get("id") or f"B{len(new_list) + 1}")
                new_list.append(it2)

    return new_list


def _localization_score(metrics: Dict[str, float]) -> float:
    return (
        0.7 * float(metrics.get("bbox_iou_hit", 0.0))
        + 0.3 * float(metrics.get("bbox_l1_hit", 0.0))
    )


def _relative_improve_ratio(prev_metrics: Dict[str, float], final_metrics: Dict[str, float]) -> Tuple[float, float, float]:
    """Use continuous IoU/L1 localization improvements for refine reward."""
    prev_iou = float(prev_metrics.get("mean_iou", 0.0))
    final_iou = float(final_metrics.get("mean_iou", 0.0))
    iou_gain_ratio = max(0.0, (final_iou - prev_iou) / max(1.0 - prev_iou, IMPROVE_EPS))

    prev_l1 = float(prev_metrics.get("mean_bbox_l1", 0.0))
    final_l1 = float(final_metrics.get("mean_bbox_l1", 0.0))
    prev_l1_quality = _bbox_l1_quality(prev_l1)
    final_l1_quality = _bbox_l1_quality(final_l1)
    l1_gain_ratio = max(0.0, (final_l1_quality - prev_l1_quality) / max(1.0 - prev_l1_quality, IMPROVE_EPS))

    prev_cont_score = _continuous_localization_score(prev_metrics)
    final_cont_score = _continuous_localization_score(final_metrics)
    gain_ratio = max(0.0, (final_cont_score - prev_cont_score) / max(1.0 - prev_cont_score, IMPROVE_EPS))

    return gain_ratio, iou_gain_ratio, l1_gain_ratio


def _bbox_l1_quality(mean_bbox_l1: float) -> float:
    return max(0.0, 1.0 - min(float(mean_bbox_l1), float(CANVAS_SIZE)) / float(CANVAS_SIZE))


def _continuous_localization_score(metrics: Dict[str, float]) -> float:
    return (
        0.7 * float(metrics.get("mean_iou", 0.0))
        + 0.3 * _bbox_l1_quality(float(metrics.get("mean_bbox_l1", CANVAS_SIZE)))
    )


# ============================================================
# 3. Math & Geometry (Integer Pixel: +1)
# ============================================================
def _batch_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    inter_w = np.maximum(0.0, xB - xA + 1)
    inter_h = np.maximum(0.0, yB - yA + 1)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area2 = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = area1 + area2.T - inter_area
    return inter_area / np.maximum(union, 1e-6)

def _clamp_box_abs(b):
    hi = CANVAS_SIZE - 1
    return [
        int(min(hi, max(0, round(float(b[0]))))),
        int(min(hi, max(0, round(float(b[1]))))),
        int(min(hi, max(0, round(float(b[2]))))),
        int(min(hi, max(0, round(float(b[3]))))),
    ]

def _clamp_point_abs(p):
    hi = CANVAS_SIZE - 1
    return [
        int(min(hi, max(0, round(float(p[0]))))),
        int(min(hi, max(0, round(float(p[1]))))),
    ]

def _is_valid_box4(b):
    try:
        return (
            isinstance(b, (list, tuple)) and len(b) == 4 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b)
        )
    except:
        return False

def _is_valid_point2(p):
    try:
        return (
            isinstance(p, (list, tuple)) and len(p) == 2 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in p)
        )
    except:
        return False

def _extract_gt_arrays(gt_data: List[Dict]):
    gt_boxes, gt_points, gt_has_point = [], [], []
    for it in gt_data:
        if not isinstance(it, dict):
            continue
        b = it.get("bbox_2d")
        p = it.get("point_2d")
        if _is_valid_box4(b):
            gt_boxes.append([float(x) for x in b])
            if _is_valid_point2(p):
                gt_points.append([float(x) for x in p])
                gt_has_point.append(True)
            else:
                gt_points.append([0.0, 0.0])
                gt_has_point.append(False)
    return (
        np.array(gt_boxes, dtype=float),
        np.array(gt_points, dtype=float),
        np.array(gt_has_point, dtype=bool),
    )

def _extract_pred_arrays(pred_list: List[Dict]):
    pred_boxes, pred_points = [], []
    for it in pred_list:
        if not isinstance(it, dict):
            continue
        b = it.get("bbox_2d")
        p = it.get("point_2d")
        if _is_valid_box4(b):
            pred_boxes.append([float(x) for x in b])
            if _is_valid_point2(p):
                pred_points.append([float(x) for x in p])
            else:
                pred_points.append([0.0, 0.0])
    return np.array(pred_boxes, dtype=float), np.array(pred_points, dtype=float)

def _box_center(box: List[float]) -> List[float]:
    return [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]

def _extract_boxes_points_for_metrics(items: List[Dict], is_gt: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    boxes, points = [], []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        b = it.get("bbox_2d")
        p = it.get("point_2d")
        if not _is_valid_box4(b):
            continue
        box = [float(x) for x in b]
        boxes.append(box)
        if _is_valid_point2(p):
            points.append([float(p[0]), float(p[1])])
        else:
            points.append(_box_center(box))
    return np.array(boxes, dtype=float), np.array(points, dtype=float)

def _localization_metrics(items: List[Dict], gt_data: List[Dict]) -> Dict[str, float]:
    pred_boxes, pred_points = _extract_boxes_points_for_metrics(items)
    gt_boxes, gt_points = _extract_boxes_points_for_metrics(gt_data, is_gt=True)
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0:
        hit = 1.0 if n_pred == 0 else 0.0
        return {
            "bbox_iou_hit": hit,
            "bbox_l1_hit": hit,
            "point_hit": hit,
            "mean_iou": hit,
            "mean_bbox_l1": 0.0 if n_pred == 0 else float(CANVAS_SIZE),
        }
    if n_pred == 0:
        return {
            "bbox_iou_hit": 0.0,
            "bbox_l1_hit": 0.0,
            "point_hit": 0.0,
            "mean_iou": 0.0,
            "mean_bbox_l1": float(CANVAS_SIZE),
        }

    iou_mat = _batch_iou(pred_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
    denom = float(max(n_pred, n_gt, 1))

    iou_hits = 0
    bbox_l1_hits = 0
    point_hits = 0
    matched_iou_sum = 0.0
    matched_bbox_l1_sum = 0.0
    for r, c in zip(row_ind, col_ind):
        pair_iou = float(iou_mat[r, c])
        matched_iou_sum += pair_iou
        if pair_iou >= BBOX_IOU_THRESHOLD:
            iou_hits += 1

        edge_l1 = float(np.mean(np.abs(pred_boxes[r] - gt_boxes[c])))
        matched_bbox_l1_sum += edge_l1
        if edge_l1 <= BBOX_EDGE_L1_THRESHOLD:
            bbox_l1_hits += 1

        point_l1 = float(np.abs(pred_points[r] - gt_points[c]).sum())
        if point_l1 <= POINT_L1_THRESHOLD:
            point_hits += 1

    return {
        "bbox_iou_hit": float(iou_hits) / denom,
        "bbox_l1_hit": float(bbox_l1_hits) / denom,
        "point_hit": float(point_hits) / denom,
        "mean_iou": matched_iou_sum / denom,
        "mean_bbox_l1": (matched_bbox_l1_sum + (denom - len(row_ind)) * float(CANVAS_SIZE)) / denom,
    }


# ============================================================
# 4. Reward Components
# ============================================================
def _calculate_step_accuracy(items: List[Dict], gt_data: List[Dict]) -> float:
    """Helper to calculate accuracy for a single state."""
    pred_boxes, pred_points = _extract_pred_arrays(items)
    gt_boxes, gt_points, gt_has_point = _extract_gt_arrays(gt_data)

    if len(gt_boxes) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0

    iou_mat = _batch_iou(pred_boxes, gt_boxes)
    cost_mat = np.zeros_like(iou_mat)

    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            score = iou_mat[i, j]
            if gt_has_point[j]:
                dist = np.linalg.norm(pred_points[i] - gt_points[j])
                pt_score = max(0.0, 1.0 - dist / 50.0)
                score += pt_score
                cost_mat[i, j] = 2.0 - score
            else:
                cost_mat[i, j] = 1.0 - score

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    total_reward = 0.0
    for r, c in zip(row_ind, col_ind):
        s = iou_mat[r, c]
        if gt_has_point[c]:
            dist = np.linalg.norm(pred_points[r] - gt_points[c])
            pt_score = max(0.0, 1.0 - dist / 50.0)
            s += pt_score
            s /= 2.0
        total_reward += s

    denominator = max(len(pred_boxes), len(gt_boxes), 1)
    return float(total_reward / denominator)

def _accuracy_reward(states: List, gt_data: List[Dict]) -> float:
    """Legacy wrapper for component logging (uses final state)."""
    if not states:
        return 0.0
    return _calculate_step_accuracy(states[-1], gt_data)

def _delta_reward(states: List, gt_data: List[Dict]) -> float:
    if len(states) < 2:
        return 0.0

    prev_score = _calculate_step_accuracy(states[-2], gt_data)
    curr_score = _calculate_step_accuracy(states[-1], gt_data)
    return max(0.0, curr_score - prev_score)

def _efficiency_reward(states: List, gt_data: List[Dict]) -> float:
    if not states:
        return 0.0

    threshold = 0.7
    for t, items in enumerate(states):
        score = _calculate_step_accuracy(items, gt_data)
        if score >= threshold:
            return 1.0 / (1.0 + 0.5 * t)
    return 0.0

def _cardinality_reward(states: List, gt_data: List[Dict]) -> float:
    if not states:
        return 0.0
    final_items = states[-1]
    n_pred = len(final_items)
    n_gt = len(gt_data)
    diff = abs(n_pred - n_gt)
    if diff == 0:
        return 1.0
    return max(0.0, 1.0 - 0.2 * diff)
