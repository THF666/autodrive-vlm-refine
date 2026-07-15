# 数据契约

## 1. 场景统计 TXT

允许的内容只有场景行、空行和两条可选汇总：

```text
00000000000000000000000000000001<TAB>13
00000000000000000000000000000002<TAB>16

总场景数: 2
总图像数: 29
```

真实文件不要提交。切分器会拒绝重复 scene id、非正数计数、无法识别的行以及错误的汇总。

## 2. 单帧标注文件名

```text
{32位scene_uuid}_{32位frame_uuid}.json
```

切分器会递归扫描 annotation 目录，并逐场景比较“TXT 计数”和“实际 JSON 数”。任一不一致都会停止，避免静默漏图。

## 3. 原始标注 JSON

归一化工具默认支持下列常见字段：

```json
{
  "image": "relative/or/absolute/image.jpg",
  "bbox": [[x1, y1, x2, y2], [x1, y1, x2, y2]]
}
```

也自动尝试 `boxes`、`bboxes`、`annotations` 以及 `image_path`、`img_name`、`imagePath`、`filename`。嵌套字段用参数指定，例如：

```bash
--boxes-key data.annotations --image-key data.image_path
```

如果 `annotations` 是对象列表，每个对象必须含 `bbox`、`bbox_2d` 或 `box`。若原始框是 `[x, y, width, height]`，传 `--box-format xywh`。

## 4. 归一化 JSONL

所有训练阶段的内部标准是 840×840 图像和整数 `xyxy`：

```json
{
  "id": "sceneuuid_frameuuid",
  "scene_id": "sceneuuid",
  "image": "/absolute/private/path/prepared/scene/frame.jpg",
  "img_name": "scene/frame.jpg",
  "boxes": [[84, 100, 240, 300]],
  "bbox": [[84, 100, 240, 300]],
  "problem": "target object",
  "width": 840,
  "height": 840
}
```

归一化阶段会真正生成 840×840 私有 JPEG，并按原图尺寸缩放 box。不能只让训练框架动态缩图，因为 Refine 输出中的数值必须和图像坐标系一致。

## 5. Stage 2 ms-swift JSONL

生成器输出官方标准的 `messages + images`：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>Locate ... previous predictions ..."},
    {"role": "assistant", "content": "<think>...</think><answer>{...}</answer>"}
  ],
  "images": ["/absolute/private/path/frame.jpg"],
  "metadata": {"scenario": "mixed"}
}
```

纠错格式保持与 Stage 3 reward 一致：

- 已有正确目标：对 `bbox` 和 `point` 输出 delta。
- 误检/重复框：`action: "delete"`。
- 漏检：放入 `new_items`，同时给 `bbox_2d` 和框中心 `point_2d`。

每个生成样本都会在写出前回放 action；只有最终 boxes 与 GT 完全一致才会通过。

## 6. Stage 3 hard-case 输入

`select_hard_cases.py` 的输入是“训练集推理结果与 GT 合并后的 JSONL”，至少包含：

```json
{
  "id": "...",
  "scene_id": "...",
  "image": "/private/path/frame.jpg",
  "boxes": [[100, 120, 220, 280]],
  "problem": "target object",
  "metrics": {"f1": 0.25}
}
```

脚本强制传入 `train_scenes.txt` allow-list；发现任何 test scene 会直接失败。
