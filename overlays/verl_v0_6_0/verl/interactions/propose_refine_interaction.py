# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import logging
import re
import os
import uuid
import socket
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from .base import BaseInteraction

logger = logging.getLogger(__name__)

# =========================
# Constants & Paths
# =========================
VIZ_DIR = os.getenv(
    "PR_VIZ_DIR",
    os.path.join(os.getcwd(), "logs", "propose_refine_viz"),
)

MAX_USER_TEXT = 600
MAX_FINAL_PROMPT = 1600

# 给模型看的“标准 refine JSON 格式”
REFINE_FORMAT_JSON_STR = (
    '{"per_bbox":['
    '{"id":"B1","action":{"bbox":[dleft,dtop,dright,dbottom],"point":[dX,dY]}},'
    '{"id":"B2","action":"delete"}'
    '],'
    '"new_items":[{"bbox_2d":[x1,y1,x2,y2],"point_2d":[px,py]}]}'
)

QUESTION_RES = [
    re.compile(r'please\s+find\s+"([^"]+)"\s+with\s+bbox(?:es|s)?\s+and\s+points', re.IGNORECASE),
    re.compile(r'locate\s+"([^"]+)"', re.IGNORECASE),
    re.compile(r'find\s+"([^"]+)"', re.IGNORECASE),
]

EXAMPLE_ANSWER = (
    '[{"bbox_2d":[x1,y1,x2,y2],"point_2d":[x3,y3]},'
    '{"bbox_2d":[x1,y1,x2,y2],"point_2d":[x3,y3]}]'
)

# =========================
# Helpers
# =========================
try:
    from omegaconf import DictConfig  # type: ignore
except Exception:  # pragma: no cover
    DictConfig = None  # type: ignore


def _fs_slug(s: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", s or "")
    return slug[:max_len] if slug else "req"


def _make_unique_path(request_id: str, round_num: int) -> str:
    os.makedirs(VIZ_DIR, exist_ok=True)
    tag = _fs_slug(request_id)
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    uid = uuid.uuid4().hex[:8]
    fname = f"PR_{tag}_r{round_num}_{host}_p{pid}_{ts}_{uid}.txt"
    return os.path.join(VIZ_DIR, fname)


def _format_conversation_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, m in enumerate(messages, 1):
        role = m.get("role", "unknown")
        lines.append(f"===== [{i}] role: {role} =====")
        content = m.get("content", "")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for j, seg in enumerate(content, 1):
                if isinstance(seg, dict):
                    typ = seg.get("type")
                    if typ == "text":
                        lines.append(seg.get("text", ""))
                    elif typ == "image":
                        lines.append(f"[image #{j}]")
                    elif typ == "video":
                        lines.append(f"[video #{j}]")
                    else:
                        lines.append(json.dumps(seg, ensure_ascii=False))
                else:
                    lines.append(str(seg))
        elif isinstance(content, dict):
            lines.append(json.dumps(content, ensure_ascii=False, indent=2))
        else:
            lines.append(str(content))
        tool_calls = m.get("tool_calls")
        if tool_calls:
            try:
                lines.append("---- tool_calls ----")
                lines.append(json.dumps(tool_calls, ensure_ascii=False, indent=2))
            except Exception:
                pass
        lines.append("")
    return "\n".join(lines)


def _cfg_get(cfg: Any, keys: List[str], default: Optional[int] = None) -> Optional[int]:
    if cfg is None:
        return default
    for k in keys:
        try:
            if DictConfig is not None and isinstance(cfg, DictConfig) and k in cfg:
                return cfg[k]
        except Exception:
            pass
        if isinstance(cfg, dict) and k in cfg:
            return cfg[k]
        v = getattr(cfg, k, None)
        if v is not None:
            return v
    return default


def _first_user_text(messages: List[Dict[str, Any]], max_len: int = MAX_USER_TEXT) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            return content[:max_len]
        if isinstance(content, list):
            for seg in content:
                if isinstance(seg, dict) and seg.get("text"):
                    txt = str(seg["text"]).strip()
                    if txt:
                        return txt[:max_len]
            txt = "".join(
                (seg.get("text") or "")
                for seg in content
                if isinstance(seg, dict) and seg.get("text")
            ).strip()
            if txt:
                return txt[:max_len]
        return ""
    return ""


def _extract_question_from_first_user(messages: List[Dict[str, Any]]) -> str:
    txt = _first_user_text(messages)
    if not txt:
        return "the referred object"
    for pat in QUESTION_RES:
        m = pat.search(txt)
        if m:
            return m.group(1).strip()
    m2 = re.search(r'"([^"]+)"', txt)
    if m2:
        return m2.group(1).strip()
    return txt[:120]


def _is_valid_abs_box(box: Any) -> bool:
    if not (isinstance(box, list) and len(box) == 4):
        return False
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return False
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def _fix_box_order(box: List[float]) -> List[float]:
    if not (isinstance(box, list) and len(box) == 4):
        return box
    x1, y1, x2, y2 = box
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _as_point2_safe(pt: Any) -> Optional[List[int]]:
    if not (isinstance(pt, list) and len(pt) == 2):
        return None
    try:
        return [int(round(float(pt[0]))), int(round(float(pt[1])))]
    except Exception:
        return None


_TAG_RE = {
    "think": re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE),
    "answer": re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE),
}

def extract_tag_text(text: str, tag: str) -> Optional[str]:
    m = _TAG_RE[tag].search(text or "")
    return m.group(1).strip() if m else None


@dataclass
class FormatCheckResult:
    is_valid: bool
    errors: List[str]
    parsed: Optional[List[Dict[str, Any]]] = None
    normalized_text: Optional[str] = None


def _maybe_strip_one_outer_list(ans: Any) -> Any:
    if isinstance(ans, list) and len(ans) == 1 and isinstance(ans[0], list):
        return ans[0]
    return ans


def _maybe_wrap_single_object(ans: Any) -> Any:
    if isinstance(ans, dict):
        return [ans]
    return ans


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _fix_xyxy(box: List[int]) -> List[int]:
    x1, y1, x2, y2 = box
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _as_int_list(vals: Any, n: int) -> Tuple[bool, Optional[List[int]], Optional[str]]:
    if not (isinstance(vals, list) and len(vals) == n):
        return False, None, f"should be a list of length {n}, got {vals!r}"
    out: List[int] = []
    for v in vals:
        if not isinstance(v, (int, float)):
            return False, None, f"contains non-numeric value {v!r}"
        out.append(int(round(float(v))))
    return True, out, None


def _area_xyxy(box: List[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _point_inside_box(pt: List[int], box: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _box_center_int(box: List[int]) -> List[int]:
    return [int(round((box[0] + box[2]) / 2.0)), int(round((box[1] + box[3]) / 2.0))]


def _validate_answer_objs(objs: Any) -> Tuple[bool, List[str], Optional[List[Dict[str, Any]]]]:
    errors: List[str] = []
    if not isinstance(objs, list) or len(objs) == 0:
        return False, ["`answer` must be a non-empty JSON array."], None

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(objs):
        path = f"[B{i+1}]"
        if not isinstance(item, dict):
            errors.append(f"{path} is not an object (dict).")
            continue

        if "bbox_2d" not in item:
            errors.append(f"{path} missing required key `bbox_2d`.")
        if "point_2d" not in item:
            errors.append(f"{path} missing required key `point_2d`.")
        if "bbox_2d" not in item or "point_2d" not in item:
            continue

        ok_box, box_ints, err_box = _as_int_list(item["bbox_2d"], 4)
        if not ok_box:
            errors.append(f"{path}.bbox_2d {err_box}.")
            continue

        ok_pt, pt_ints, err_pt = _as_int_list(item["point_2d"], 2)
        if not ok_pt:
            errors.append(f"{path}.point_2d {err_pt}.")
            continue

        box_fixed = _fix_xyxy(box_ints)
        if _area_xyxy(box_fixed) <= 0:
            errors.append(f"{path}.bbox_2d has zero or negative area after ordering: {box_fixed}.")
            continue

        if not _point_inside_box(pt_ints, box_fixed):
            errors.append(
                f"{path}.point_2d {pt_ints} is not inside bbox_2d {box_fixed} (inclusive)."
            )
        normalized.append({"bbox_2d": box_fixed, "point_2d": pt_ints})

    return (len(errors) == 0), errors, (normalized if len(normalized) > 0 else None)


def check_propose_output(raw_text: str) -> FormatCheckResult:
    if not raw_text or not isinstance(raw_text, str):
        return FormatCheckResult(False, ["Output is empty or not a string."])

    ans_text = extract_tag_text(raw_text, "answer")
    if ans_text is None:
        return FormatCheckResult(False, ["Missing <answer>...</answer> tag."])

    ans_text = _strip_code_fences(ans_text)
    try:
        ans_json = json.loads(ans_text)
    except Exception as e:
        return FormatCheckResult(False, [f"`answer` is not valid JSON. Parse error: {e}"])

    ans_json = _maybe_strip_one_outer_list(ans_json)
    ans_json = _maybe_wrap_single_object(ans_json)

    ok, errs, normalized = _validate_answer_objs(ans_json)
    if not ok:
        return FormatCheckResult(False, errs)

    if not normalized:
        return FormatCheckResult(False, ["All items are invalid after validation."], parsed=None, normalized_text=None)

    norm_text = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return FormatCheckResult(True, [], parsed=normalized, normalized_text=norm_text)


def build_propose_prompt(question: str) -> str:
    return (
        f'Please find "{question}" with bboxes and points.\n'
        "Compare the candidates and keep the most closely matched object(s)."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "Output the thinking process in <think>...</think> and final answer in <answer>...</answer>.\n"
        "i.e., <think> thinking process here </think>"
        f"<answer>{EXAMPLE_ANSWER}</answer>\n"
    )


def build_refine_prompt(question: str, prev_items_str: str, reasons: Optional[List[str]] = None) -> str:
    reason_txt = ""
    if reasons:
        r = " ; ".join(reasons)
        reason_txt = f"Previous reply format is incorrect: {r}\n"
    # return (
    #     f'Locate "{question}".\n'
    #     f"Your previous predictions are {prev_items_str}.\n"
    #     f"e.g. {reason_txt}"
    #     "Now Refine them using the format:"
    #     "<think> thinking process here </think>"
    #     f"<answer>{REFINE_FORMAT_JSON_STR}</answer>"
    #     "Rules:\n"
    #     "- Offsets apply to each Bi: new=[x1+dleft, y1+dtop, x2+dright, y2+dbottom].\n"
    #     "- [0,0,0,0] means no-move; for points, [0,0] means no-move.\n"
    # )
    return (
        f'Locate "{question}".\n'
        f"Your previous predictions are {prev_items_str}.\n"
        "Now Refine them using the format:"
        "<think> thinking process here </think>"
        f"<answer>{REFINE_FORMAT_JSON_STR}</answer>"
    )


def _extract_pred_raw_from_answer(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict):
                parts.append(seg.get("text", "") or "")
            else:
                parts.append(str(seg))
        text = "".join(parts)
    elif isinstance(content, dict):
        text = content.get("text", "") or ""
    else:
        return "[]"

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        m2 = re.search(r"<answer>\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if not m2:
            return "[]"
        cand = m2.group(1).strip()
    else:
        cand = m.group(1).strip()

    cand = re.sub(r"^```json\s*|\s*```$", "", cand, flags=re.IGNORECASE).strip()
    cand = re.sub(r"</think>.*$", "", cand, flags=re.IGNORECASE | re.DOTALL).strip()
    return cand or "[]"


def _as_xyxy4_safe(bbox: Any) -> Optional[List[int]]:
    if not isinstance(bbox, list):
        return None
    if len(bbox) == 1 and isinstance(bbox[0], list):
        bbox = bbox[0]
    if len(bbox) != 4:
        return None
    try:
        ints = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    return _fix_xyxy(ints)


def _propose_list_to_items(obj: Any, max_items: int = 30) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        objs = [obj]
    elif isinstance(obj, list):
        objs = obj
    else:
        return []
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(objs):
        if i >= max_items:
            break
        if not isinstance(item, dict):
            continue
        bbox = None
        if "bbox_2d" in item and isinstance(item["bbox_2d"], list):
            bbox = item["bbox_2d"]
        elif "bbox" in item and isinstance(item["bbox"], list):
            bbox = item["bbox"]
        elif "box" in item and isinstance(item["box"], list):
            bbox = item["box"]
        if bbox is None:
            continue
        int_box = _as_xyxy4_safe(bbox)
        if int_box is None:
            continue
        pt = item.get("point_2d")
        int_pt = _as_point2_safe(pt)
        if int_pt is None:
            int_pt = _box_center_int(int_box)
        out.append({"id": f"B{i+1}", "bbox": int_box, "point": int_pt})
    return out


def _apply_refine_to_items(
    prev_items: List[Dict[str, Any]],
    refine_obj: Dict[str, Any],
    max_items: int = 120,
) -> List[Dict[str, Any]]:
    id2: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for it in prev_items:
        bid = it.get("id")
        bb = it.get("bbox")
        pt = it.get("point")
        if not bid or not isinstance(bb, list):
            continue
        clean_bb = _fix_xyxy([int(round(float(v))) for v in bb])
        id2[bid] = {"bbox": clean_bb, "point": list(pt) if isinstance(pt, list) else _box_center_int(clean_bb)}
        order.append(bid)

    per_bbox = refine_obj.get("per_bbox", [])
    if isinstance(per_bbox, list):
        for act in per_bbox:
            if not isinstance(act, dict):
                continue
            bid = act.get("id")
            action = act.get("action")
            if not bid or bid not in id2:
                continue

            if action == "delete":
                id2[bid]["__deleted__"] = True
                continue

            if isinstance(action, list) and len(action) == 4:
                base = id2[bid]["bbox"]
                try:
                    dleft, dtop, dright, dbottom = [float(x) for x in action]
                except Exception:
                    continue
                new_box = [
                    int(round(base[0] + dleft)),
                    int(round(base[1] + dtop)),
                    int(round(base[2] + dright)),
                    int(round(base[3] + dbottom)),
                ]
                id2[bid]["bbox"] = _fix_xyxy(new_box)
                continue

            if isinstance(action, dict):
                bbox_d = action.get("bbox")
                if isinstance(bbox_d, list) and len(bbox_d) == 4:
                    try:
                        dleft, dtop, dright, dbottom = [float(x) for x in bbox_d]
                        base = id2[bid]["bbox"]
                        new_box = [
                            int(round(base[0] + dleft)),
                            int(round(base[1] + dtop)),
                            int(round(base[2] + dright)),
                            int(round(base[3] + dbottom)),
                        ]
                        id2[bid]["bbox"] = _fix_xyxy(new_box)
                    except Exception:
                        pass
                p_d = action.get("point")
                if isinstance(p_d, list) and len(p_d) == 2:
                    try:
                        dx, dy = float(p_d[0]), float(p_d[1])
                        cur = id2[bid].get("point")
                        if isinstance(cur, list):
                            id2[bid]["point"] = [int(round(cur[0] + dx)), int(round(cur[1] + dy))]
                    except Exception:
                        pass
                p_abs = action.get("point_abs")
                p_abs_i = _as_point2_safe(p_abs)
                if p_abs_i is not None:
                    id2[bid]["point"] = p_abs_i

    new_items: List[Dict[str, Any]] = []
    nitems = refine_obj.get("new_items", [])
    if isinstance(nitems, list):
        for it in nitems:
            b = None
            p = None
            if isinstance(it, dict):
                b = it.get("bbox_2d") or it.get("bbox")
                p = it.get("point_2d") or it.get("point")
            elif isinstance(it, list):
                if len(it) >= 1 and isinstance(it[0], list) and len(it[0]) == 4:
                    b = it[0]
                    if len(it) > 1 and isinstance(it[1], list):
                        p = it[1]
                elif len(it) == 4:
                    b = it
            else:
                continue

            b_i = _as_xyxy4_safe(b)
            if b_i is None:
                continue
            p_i = _as_point2_safe(p)
            if p_i is None:
                p_i = _box_center_int(b_i)
            new_id = f"B{len(order) + len(new_items) + 1}"
            new_items.append({"id": new_id, "bbox": b_i, "point": p_i})

    nbs = refine_obj.get("new_bboxes", [])
    if isinstance(nbs, list):
        for nb in nbs:
            b_i = _as_xyxy4_safe(nb)
            if b_i is None:
                continue
            new_id = f"B{len(order) + len(new_items) + 1}"
            new_items.append({"id": new_id, "bbox": b_i, "point": None})

    final: List[Dict[str, Any]] = []
    for bid in order:
        it = id2[bid]
        if it.get("__deleted__"):
            continue
        bb = [int(round(float(v))) for v in it["bbox"]]
        pt = it.get("point")
        pt = [int(round(float(v))) for v in pt] if isinstance(pt, list) else None
        final.append({"id": bid, "bbox": _fix_xyxy(bb), "point": pt})
    final.extend(new_items)
    renum = []
    for i, it in enumerate(final[:max_items], 1):
        renum.append({"id": f"B{i}", "bbox": it["bbox"], "point": it.get("point")})
    return renum


def _clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get("bbox")
        if _is_valid_abs_box(bb):
            int_box = [int(round(float(x))) for x in bb]
            pt = it.get("point")
            pt_i = [int(round(float(v))) for v in pt] if isinstance(pt, list) else _box_center_int(_fix_xyxy(int_box))
            cleaned.append({"id": it.get("id"), "bbox": _fix_xyxy(int_box), "point": pt_i})

    final = []
    for i, it in enumerate(cleaned):
        final.append({"id": f"B{i+1}", "bbox": it["bbox"], "point": it.get("point")})
    return final


def _looks_like_refine_list(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(it, dict) and "id" in it and "action" in it for it in obj)
    )


def _refine_is_stable_from_answer(raw_answer: Any, tol: float = 1e-6) -> bool:
    pred_raw = _extract_pred_raw_from_answer(raw_answer)
    try:
        obj = json.loads(pred_raw)
    except Exception:
        return False

    if not isinstance(obj, dict):
        return False

    new_bboxes = obj.get("new_bboxes", [])
    new_items = obj.get("new_items", [])
    if (isinstance(new_bboxes, list) and len(new_bboxes) > 0) or (isinstance(new_items, list) and len(new_items) > 0):
        return False

    per_bbox = obj.get("per_bbox", [])
    if not isinstance(per_bbox, list):
        return True

    for act in per_bbox:
        if not isinstance(act, dict):
            return False
        action = act.get("action", None)
        if action is None:
            continue
        if action == "delete":
            return False
        if isinstance(action, list) and len(action) == 4:
            try:
                if any(abs(float(x)) > tol for x in action):
                    return False
            except Exception:
                return False
        elif isinstance(action, dict):
            if "bbox" in action:
                try:
                    if any(abs(float(x)) > tol for x in action.get("bbox", [])):
                        return False
                except Exception:
                    return False
            if "point" in action:
                try:
                    if any(abs(float(x)) > tol for x in action.get("point", [])):
                        return False
                except Exception:
                    return False
            if "point_abs" in action:
                return False
        else:
            return False
    return True


def _backtrack_propose_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_one_assistant = False
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        if not seen_one_assistant:
            seen_one_assistant = True
            continue
        raw_answer = m.get("content", "") or ""
        pred_raw = _extract_pred_raw_from_answer(raw_answer)
        try:
            obj = json.loads(pred_raw)
        except Exception:
            continue
        if isinstance(obj, list) or (isinstance(obj, dict) and "bbox_2d" in obj):
            return _propose_list_to_items(obj)
    return []

# =========================
# Interaction
# =========================
class ProposeRefineInteraction(BaseInteraction):
    name = "propose_refine"

    def __init__(self, config: Any):
        super().__init__(config)
        keys = ["max_round", "max_rounds", "max_user_turns"]
        self.default_max_round = int(_cfg_get(config, keys, default=5))
        # 新增：是否在解析失败时立即终止（默认 True）
        self.terminate_on_parse_error: bool = True
        self._req_state: Dict[str, Dict[str, Any]] = {}

    async def start_interaction(self, request_id: str, **kwargs):
        keys = ["max_round", "max_rounds", "max_user_turns"]
        max_round = kwargs.get("max_round")
        if max_round is None:
            max_round = _cfg_get(kwargs, keys, default=self.default_max_round)
        max_round = int(max_round)

        self._req_state[request_id] = {
            "cur_round": 0,
            "max_round": max_round,
            "items": [],      # [{"id","bbox","point"}]
            "history": [],    # 记录每轮轨迹
            "terminated": False,
            "terminal_reason": None,
        }

    async def finalize_interaction(self, request_id: str, **kwargs):
        self._req_state.pop(request_id, None)

    async def generate_response(
        self,
        request_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        save_txt = False
        state = self._req_state.setdefault(
            request_id,
            {"cur_round": 0, "max_round": self.default_max_round, "items": [], "history": [],
             "terminated": False, "terminal_reason": None},
        )
        state["cur_round"] += 1
        cur_round = state["cur_round"]
        max_round = state["max_round"]

        last_assistant = None
        for m in reversed(messages):
            if m.get("role") == "assistant":
                last_assistant = m
                break

        question = _extract_question_from_first_user(messages)
        #print('='*50,'=messages',messages)

        current_items = state.get("items", []) or []
        in_refine_phase = len(current_items) > 0
        invalid_reasons: List[str] = []
        parse_failed = False  # 新增：解析失败标记

        items_before = copy.deepcopy(state.get("items", []))

        if last_assistant:
            raw_answer = last_assistant.get("content", "") or ""

            if not in_refine_phase:
                # === PROPOSE ===
                # Round 1 consumes the model's proposal answer. A valid proposal
                # becomes the persistent state shown to the model in the next
                # user message as B1/B2/... previous predictions.
                check = check_propose_output(raw_answer)
                if check.is_valid:
                    new_items = _propose_list_to_items(check.parsed)
                    state["items"] = _clean_items(new_items)
                else:
                    invalid_reasons = check.errors or ["Your previous response format is invalid."]
                    parse_failed = True  # 立刻判为失败
            else:
                # === REFINE ===
                # Round 2 consumes a refine action and applies it to the saved
                # proposal state. The interaction parser is intentionally more
                # tolerant than reward.py so rollout can finish cleanly, while
                # reward.py decides whether the final schema deserves credit.
                pred_raw = _extract_pred_raw_from_answer(raw_answer)
                try:
                    obj = json.loads(pred_raw)
                except Exception:
                    obj = None
                if isinstance(obj, list) and _looks_like_refine_list(obj):
                    prev_items = state.get("items", []) or _backtrack_propose_from_messages(messages)
                    refine_obj = {"per_bbox": obj, "new_bboxes": []}
                    applied = _apply_refine_to_items(prev_items, refine_obj)
                    state["items"] = _clean_items(applied)
                elif isinstance(obj, dict) and "per_bbox" in obj:
                    prev_items = state.get("items", []) or _backtrack_propose_from_messages(messages)
                    applied = _apply_refine_to_items(prev_items, obj)
                    state["items"] = _clean_items(applied)
                else:
                    invalid_reasons.append("Unable to parse a valid REFINE JSON object.")
                    parse_failed = True  # 立刻判为失败

            # 记录轨迹
            try:
                ans_inner = extract_tag_text(raw_answer, "answer") or ""
                pred_json_raw = _extract_pred_raw_from_answer(raw_answer)
                entry = {
                    "round": cur_round,
                    "raw_answer": raw_answer,
                    "answer": ans_inner,
                    "answer_json_raw": pred_json_raw,
                    "items_before": items_before,
                    "items_after": copy.deepcopy(state.get("items", [])),
                    "phase": "refine" if in_refine_phase else "propose",
                    "invalid_reasons": invalid_reasons[:],
                    "parse_failed": parse_failed,
                }
                state["history"].append(entry)
            except Exception as _e:
                state["history"].append({
                    "round": cur_round,
                    "raw_answer": "<record-failed>",
                    "answer": "<record-failed>",
                    "error": str(_e),
                    "items_before": items_before,
                    "items_after": copy.deepcopy(state.get("items", [])),
                    "phase": "refine" if in_refine_phase else "propose",
                    "invalid_reasons": invalid_reasons[:],
                    "parse_failed": parse_failed,
                })

        debug_print = os.getenv("PROPOSE_REFINE_INTERACTION_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
        if debug_print:
            print(
                f"[ProposeRefine][PRINT] request_id={request_id} round={cur_round} "
                f"items={json.dumps(state.get('items', []), ensure_ascii=False)}"
            )

        # === 决策是否终止 ===
        should_terminate = False
        final_prompt = ""

        if self.terminate_on_parse_error and parse_failed:
            should_terminate = True
            state["terminated"] = True
            state["terminal_reason"] = "parse_error"
        else:
            # 常规停止条件
            should_terminate = cur_round >= (max_round - 1)
            if (
                not should_terminate
                and cur_round > 1
                and last_assistant is not None
                and state.get("items")
            ):
                raw_answer = last_assistant.get("content", "") or ""
                if _refine_is_stable_from_answer(raw_answer, tol=1e-6):
                    should_terminate = True

        # 若不终止则下发下一轮 prompt
        if not should_terminate:
            if state.get("items"):
                # Convert internal {"bbox","point"} state back to the prompt
                # schema {"bbox_2d","point_2d"} expected by the SFT/RL format.
                current_items_show = []
                for it in state["items"]:
                    int_box = [int(round(float(v))) for v in it.get("bbox", [])]
                    pt = it.get("point")
                    pt_i = [int(round(float(v))) for v in pt] if isinstance(pt, list) else None
                    obj = {"id": it.get("id", ""), "bbox_2d": int_box}
                    if pt_i is not None:
                        obj["point_2d"] = pt_i
                    current_items_show.append(obj)
                prev_items_str = json.dumps(current_items_show, ensure_ascii=False, separators=(",", ":"))
                final_prompt = build_refine_prompt(question, prev_items_str, invalid_reasons if invalid_reasons else None)
            else:
                # 正常的 propose 首轮提示（不再重试，失败就终止）
                final_prompt = build_propose_prompt(question)

            if final_prompt and len(final_prompt) > MAX_FINAL_PROMPT:
                final_prompt = final_prompt[:MAX_FINAL_PROMPT] + " ..."

        if debug_print:
            print(
                "*******************************************Should_Terminate**************************",
                cur_round, ":", should_terminate, max_round, "; parse_failed:", parse_failed
            )

        if should_terminate:
            final_prompt = ""

            if save_txt:
                try:
                    out_path = _make_unique_path(request_id, cur_round)
                    lines: List[str] = []
                    lines.append("========== Propose-Refine Debug ==========")
                    lines.append(f"request_id: {request_id}")
                    lines.append(f"rounds    : {cur_round}")
                    lines.append(f'terminated: {state.get("terminated")}, reason={state.get("terminal_reason")}')
                    lines.append(f'question  : "{question}"')
                    lines.append("")
                    lines.append("----- Conversation (raw) -----")
                    lines.append(_format_conversation_text(messages))
                    lines.append("")
                    lines.append("----- Trajectory (per round) -----")
                    hist = state.get("history", [])
                    if not isinstance(hist, list):
                        hist = []
                    for i, rec in enumerate(hist, 1):
                        lines.append(f"[Round {i}]")
                        if isinstance(rec, dict):
                            raw_ans = rec.get("raw_answer", "")
                            if raw_ans:
                                lines.append("<<< answer raw >>>")
                                lines.append(raw_ans)

                            ajr = rec.get("answer_json_raw", None)
                            lines.append("answer_json_raw:")
                            if ajr is not None:
                                try:
                                    ajr_parsed = json.loads(ajr)
                                    lines.append(json.dumps(ajr_parsed, ensure_ascii=False, indent=2))
                                except Exception:
                                    lines.append(str(ajr))
                            else:
                                lines.append("<none>")

                            b_before = rec.get("items_before", None)
                            if b_before is not None:
                                try:
                                    lines.append("items_before:")
                                    lines.append(json.dumps(b_before, ensure_ascii=False, indent=2))
                                except Exception:
                                    lines.append("items_before: <serialize-failed>")

                            b_after = rec.get("items_after", None)
                            if b_after is not None:
                                try:
                                    lines.append("items_after:")
                                    lines.append(json.dumps(b_after, ensure_ascii=False, indent=2))
                                except Exception:
                                    lines.append("items_after: <serialize-failed>")

                            reasons = rec.get("invalid_reasons", [])
                            if reasons:
                                lines.append("invalid_reasons:")
                                lines.append(json.dumps(reasons, ensure_ascii=False, indent=2))
                            lines.append(f"parse_failed: {rec.get('parse_failed')}")
                        lines.append("")

                    lines.append("----- Final Items -----")
                    try:
                        lines.append(json.dumps(state.get("items", []), ensure_ascii=False, indent=2))
                    except Exception:
                        lines.append("<final items serialize-failed>")

                    with open(out_path, "x", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    print(f"[ProposeRefine][TRACE] saved to {out_path}")
                except FileExistsError:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write("\n\n[WARN] Filename collision, appended.\n")
                    print(f"[ProposeRefine][TRACE] appended to {out_path}")
                except Exception as _e:
                    print(f"[ProposeRefine][TRACE][SAVE_FAILED] err={_e}")

            # 重要：终止时清理状态
            state["terminated"] = True
            if state.get("terminal_reason") is None:
                state["terminal_reason"] = "normal_stop"
            # 将必要元信息通过 meta 返回，给奖励函数使用
            meta = {
                "round": cur_round,
                "terminated": True,
                "terminal_reason": state.get("terminal_reason"),
                "format_errors": invalid_reasons,
                "items": copy.deepcopy(state.get("items", [])),
            }
            self._req_state.pop(request_id, None)
            return True, "", 0.0, meta

        # 未终止路径：把本轮 meta 返回（供奖励函数按需使用）
        return False, final_prompt, 0.0, {
            "round": cur_round,
            "terminated": False,
            "format_errors": invalid_reasons,
            "items": copy.deepcopy(state.get("items", [])),
        }
