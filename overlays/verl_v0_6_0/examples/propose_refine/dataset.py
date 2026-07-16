import copy
import json
import logging
import os
import re
import traceback
from io import BytesIO
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image as PILImage
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from datasets import load_dataset, load_from_disk

from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__name__)
EXAMPLE_ANSWER = (
    '[{"bbox_2d":[x1,y1,x2,y2],"point_2d":[x3,y3]},'
    '{"bbox_2d":[x1,y1,x2,y2],"point_2d":[x3,y3]}]'
)

user_prompt = (
    "<image>"
    'Please find "{Question}" with bboxes and points.\n'
    "Compare the candidates and keep the most closely matched object(s)."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
    "Output the thinking process in <think>...</think> and final answer in <answer>...</answer>.\n"
    "i.e., <think> thinking process here </think>"
    f"<answer>{EXAMPLE_ANSWER}</answer>\n"
)

REFINE_FORMAT_JSON_STR = (
    '{"per_bbox":['
    '{"id":"B1","action":{"bbox":[dleft,dtop,dright,dbottom],"point":[dX,dY]}},'
    '{"id":"B2","action":"delete"}'
    '],'
    '"new_items":[{"bbox_2d":[x1,y1,x2,y2],"point_2d":[px,py]}]}'
)


def _proposal_for_prompt(proposal) -> list[dict]:
    if proposal is None:
        return []
    if isinstance(proposal, str):
        proposal = json.loads(proposal)
    if not isinstance(proposal, list):
        raise TypeError(f"proposal must be a list or JSON list string, got {type(proposal)!r}")

    out = []
    for i, item in enumerate(proposal):
        if not isinstance(item, dict):
            raise TypeError(f"proposal item {i} must be dict, got {type(item)!r}")
        box = item.get("bbox_2d", item.get("bbox"))
        if not (isinstance(box, list) and len(box) == 4):
            raise ValueError(f"proposal item {i} missing bbox_2d/bbox: {item!r}")
        obj = {
            "id": str(item.get("id") or f"B{i + 1}"),
            "bbox_2d": [int(round(float(v))) for v in box],
        }
        point = item.get("point_2d", item.get("point"))
        if point is not None:
            if not (isinstance(point, list) and len(point) == 2):
                raise ValueError(f"proposal item {i} has invalid point_2d/point: {item!r}")
            obj["point_2d"] = [int(round(float(point[0]))), int(round(float(point[1])))]
        else:
            x1, y1, x2, y2 = obj["bbox_2d"]
            obj["point_2d"] = [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]
        out.append(obj)
    if not out:
        raise ValueError("proposal is empty after normalization.")
    return out


def _build_refine_prompt(question: str, proposal: list[dict]) -> str:
    prev_json = json.dumps(proposal, ensure_ascii=False, separators=(",", ":"))
    return (
        "<image>"
        f'Locate "{question}".\n'
        f"Your previous predictions are {prev_json}.\n"
        "Now Refine them using the format:"
        "<think> thinking process here </think>"
        f"<answer>{REFINE_FORMAT_JSON_STR}</answer>"
    )


class CustomRLHFDataset(RLHFDataset):
    """
    关键改动
    1) __getitem__ 构造 propose-refine raw_prompt + reward/interaction 元信息
    2) 兼容当前 verl trainer，仍返回 input_ids/attention_mask/position_ids
    3) 统一把 image_key 规范成 list，避免 len(PngImageFile) 报错
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None

        self.use_shm = config.get("use_shm", False)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)

        data_file = self.original_data_files[0]
        if isinstance(data_file, str) and os.path.isdir(data_file):
            loaded = load_from_disk(data_file)
            self.dataframe = loaded["train"] if hasattr(loaded, "keys") and "train" in loaded else loaded
        else:
            self.dataframe = load_dataset(data_file)["train"]

        # 如果你确实需要过滤过长样本，可以打开
        # self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _normalize_images_field(self, row_dict: dict) -> None:
        """
        保证 row_dict[self.image_key] 是 list
        支持几种常见格式
        1) PIL.Image 或 PngImageFile
        2) {"bytes": ...} 这种 dict
        3) 已经是 list
        """
        if self.image_key not in row_dict or row_dict[self.image_key] is None:
            return

        img_obj = row_dict[self.image_key]

        # 已经是 list
        if isinstance(img_obj, list):
            return

        # PIL Image 或 ImageFile
        if isinstance(img_obj, PILImage.Image) or img_obj.__class__.__name__.endswith("ImageFile"):
            row_dict[self.image_key] = [img_obj]
            return

        # bytes dict
        if isinstance(img_obj, dict):
            row_dict[self.image_key] = [img_obj]
            return

        # 兜底
        row_dict[self.image_key] = [img_obj]

    def __getitem__(self, item):
        row_dict: dict = dict(self.dataframe[item])

        # 原始 question 文本，确保是字符串
        q = row_dict.get(self.prompt_key, "")
        if not isinstance(q, str):
            q = str(q)
        q = q.strip()

        proposal_key = self.config.get("proposal_key", "proposal")
        proposal_for_reward = None
        if proposal_key in row_dict and row_dict.get(proposal_key) not in (None, "", []):
            # Optional one-turn-refine mode:
            # if the RL row already provides a proposal, skip the model-generated
            # proposal stage and ask directly for a refine action. This mirrors
            # the cold-start SFT task and lets reward replay start from
            # extra_info["proposal"].
            proposal_for_reward = _proposal_for_prompt(row_dict.get(proposal_key))
            prompt_text = _build_refine_prompt(q, proposal_for_reward)
            data_source_default = "visionreasoner/refine_single"
        else:
            # Default RL mode:
            # first ask the model to propose boxes/points. The interaction layer
            # will parse that first assistant turn and send the second refine
            # prompt inside the same rollout trajectory.
            prompt_text = user_prompt.replace("{Question}", q)
            data_source_default = "visionreasoner/propose_refine"

        # 把 prompt_key 改成 messages，且含一个 <image> 占位符
        row_dict[self.prompt_key] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt_text,
            },
        ]

        # 关键修复 让 images 一定是 list，否则 _build_messages 里会 len(images) 崩
        self._normalize_images_field(row_dict)

        # 让 RLHFDataset._build_messages 把 <image> 替换成 {"type":"image","image":...}
        # 注意 _build_messages 会 pop(images/videos)，这是预期行为
        raw_messages = self._build_messages(row_dict)
        row_dict["raw_prompt"] = raw_messages

        # 训练侧 reward 需要的字段，你原来这样写，保持
        row_dict["reward_model"] = {"ground_truth": row_dict.get("solution", None)}

        if "data_source" not in row_dict or row_dict["data_source"] is None:
            row_dict["data_source"] = data_source_default

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        row_dict["extra_info"].setdefault("index", 0)
        row_dict["extra_info"].setdefault("tools_kwargs", {})
        if proposal_for_reward is not None:
            # reward.py uses this as the initial state for single-refine replay.
            row_dict["extra_info"]["proposal"] = proposal_for_reward
            row_dict["extra_info"]["proposal_key"] = proposal_key
        else:
            # max_round=3 means:
            # round 1 parses proposal and sends the refine prompt;
            # round 2 parses refine and terminates the trajectory.
            row_dict["extra_info"].setdefault("interaction_kwargs", {"name": "propose_refine", "max_round": 3})

        row_dict["index"] = row_dict["extra_info"]["index"]
        row_dict["tools_kwargs"] = row_dict["extra_info"]["tools_kwargs"]
        if "interaction_kwargs" in row_dict["extra_info"]:
            row_dict["interaction_kwargs"] = row_dict["extra_info"]["interaction_kwargs"]

        model_inputs = {}
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                raw_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                raw_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        return row_dict

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """
        可选过滤逻辑，修复你原实现里 row_dict 未定义等问题
        这里仅用 processor 或 tokenizer 计算 token 长度
        """
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        def _build_doc_messages(doc: dict) -> list[dict]:
            q = doc.get(prompt_key, "")
            if not isinstance(q, str):
                q = str(q)
            q = q.strip()
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": user_prompt.replace("{Question}", q),
                },
            ]

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image

            def doc2len(doc) -> int:
                try:
                    doc = dict(doc)

                    doc[prompt_key] = _build_doc_messages(doc)

                    img_obj = doc.get(image_key, None)
                    if img_obj is None:
                        images = None
                    else:
                        if isinstance(img_obj, list):
                            img_list = img_obj
                        elif isinstance(img_obj, PILImage.Image) or img_obj.__class__.__name__.endswith("ImageFile"):
                            img_list = [img_obj]
                        elif isinstance(img_obj, dict):
                            img_list = [img_obj]
                        else:
                            img_list = [img_obj]

                        images = []
                        for it in img_list:
                            if isinstance(it, PILImage.Image) or it.__class__.__name__.endswith("ImageFile"):
                                images.append(process_image(it))
                            elif isinstance(it, dict) and "bytes" in it:
                                images.append(process_image(PILImage.open(BytesIO(it["bytes"])).convert("RGB")))
                            else:
                                # 如果不是可处理类型，就跳过该样本
                                return self.max_prompt_length + 1

                    # 不处理视频
                    doc.pop(video_key, None)

                    # 使用 RLHFDataset 的替换逻辑需要 processor，直接模拟 apply_chat_template
                    messages = doc[prompt_key]
                    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    ids = processor(text=[raw_prompt], images=images, return_tensors="pt")["input_ids"][0]
                    return int(ids.numel())
                except Exception:
                    traceback.print_exc()
                    return self.max_prompt_length + 1

        else:
            def doc2len(doc) -> int:
                try:
                    messages = _build_doc_messages(doc)
                    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    return len(tokenizer.encode(raw_prompt, add_special_tokens=False))
                except Exception:
                    traceback.print_exc()
                    return self.max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe
