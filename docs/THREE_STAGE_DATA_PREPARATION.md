# 三阶段数据准备与 box-only 标注转换手册

本文说明三阶段分别需要什么数据、如何把公司现有的纯 2D box 标注转换成训练格式，以及如何安全地使用公司大模型 API 辅助构造数据。

当前基线统一使用 `Qwen2.5-VL-7B-Instruct`，图像统一处理为 840×840，坐标统一使用绝对整数 `xyxy = [x1, y1, x2, y2]`。虽然业务只需要 box，Seg-Zero 和现有 Refine reward 仍要求 `point_2d`；它由 box 中心自动计算，不需要新增人工标注。

## 1. 总体数据流与隔离原则

```text
原始图像 + box-only JSON
        │
        ├─ 按 scene UUID 做 90/10 切分
        │      ├─ train：约 51693 张，只供训练和 hard-case 挖掘
        │      └─ test：约 5744 张，只供阶段性最终评测
        │
        ├─ 统一缩放图像和 box 到 840×840
        │
        ├─ Stage 1：GT box → bbox_2d + 中心点 → HF Dataset → Proposal GRPO
        │
        ├─ Stage 2：GT + proposal → delta/delete/new_items → ms-swift JSONL → Refine SFT
        │
        └─ Stage 3：train hard cases → HF Dataset → 两轮 Propose→Refine GRPO
```

约 5.17 万张业务 train 全部进入 Stage 1。Stage 2 的 20k 和 Stage 3 的 8k 是专项训练子集；它们只能从 train 及公开数据产生，不能使用 test。

所有业务图像、UUID、JSONL、API 输出、缓存、权重和日志都保存在仓库外的公司私有磁盘。公开 GitHub 只保存代码和无敏感信息的模板。

## 2. 原始 box-only 数据要求

推荐每张图像对应一个 JSON：

```json
{
  "image": "relative/path/frame.jpg",
  "bbox": [
    [120, 80, 360, 420],
    [510, 210, 690, 480]
  ]
}
```

文件名必须能关联场景：

```text
{32位scene_uuid}_{32位frame_uuid}.json
```

当前工具也支持：

- box 字段：`bbox`、`boxes`、`bboxes`、`annotations`。
- image 字段：`image`、`image_path`、`img_name`、`imagePath`、`filename`。
- box 格式：`xyxy` 或 `xywh`。
- 嵌套字段：通过 `--boxes-key data.annotations` 等 dotted key 指定。

如果原始格式是 `xywh = [x, y, width, height]`，先转换：

```text
x1 = x
y1 = y
x2 = x + width
y2 = y + height
```

## 3. 统一到 840×840 坐标系

当前归一化工具把图像直接缩放为 840×840，不做 letterbox。对于原图宽高 `W, H`：

```text
x' = round(x × 840 / W)
y' = round(y × 840 / H)
```

然后把坐标限制到 `[0, 840]`，并保证 `x2 > x1`、`y2 > y1`。

中心点由 box 确定：

```text
point_2d = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
```

例如：

```text
bbox_2d = [100, 120, 220, 280]
point_2d = [160, 200]
```

该 point 只是与现有 Seg-Zero/Refine 代码兼容的派生字段，不是新的业务标签。以后若把 reward 改成真正的 bbox-only，才可以从 prompt、dataset、interaction 和 reward 中同时删除 point。

先建立场景切分：

```bash
python -m tools.data.build_scene_split \
  --scene-stats /private/raw/scene_stats.txt \
  --annotation-dir /private/raw/annotations \
  --output-dir /private/work/split_v1 \
  --test-ratio 0.10 \
  --seed 20250715
```

归一化 train：

```bash
python -m tools.data.normalize_annotations \
  --annotation-dir /private/raw/annotations \
  --annotation-manifest /private/work/split_v1/train_annotations.txt \
  --image-root /private/raw/images \
  --prepared-image-dir /private/work/images_840/train \
  --output-jsonl /private/work/normalized/train.jsonl \
  --box-format xyxy \
  --problem 'target object'
```

输出的内部标准 JSONL：

```json
{
  "id": "sceneuuid_frameuuid",
  "scene_id": "sceneuuid",
  "image": "/private/work/images_840/train/scene/frame.jpg",
  "img_name": "scene/frame.jpg",
  "boxes": [[100, 120, 220, 280]],
  "bbox": [[100, 120, 220, 280]],
  "problem": "target object",
  "width": 840,
  "height": 840
}
```

必须先随机抽取至少 20 张图，人工把缩放后的 box 画回图像检查。若改用等比例缩放加 letterbox，必须同时修改图像变换和 box 变换，不能与当前直接拉伸方式混用。

## 4. Stage 1：Proposal GRPO 数据

### 4.1 需要的字段

Stage 1 使用 Hugging Face `DatasetDict` 的 `train` split，核心字段为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 样本 ID |
| `problem` | string | 目标描述，例如 `target object` |
| `solution` | string | box+中心点组成的 JSON 字符串 |
| `image` | HF Image | 840×840 图像 |
| `scene_id` | string | 可选，用于审计来源 |

只有一个 box 时：

```json
{
  "id": "sample-001",
  "problem": "target object",
  "solution": "[{\"bbox_2d\":[100,120,220,280],\"point_2d\":[160,200]}]",
  "image": "<HF Image>"
}
```

多个目标时，`solution` 列表里放多个对象。顺序不作为检测语义，reward 会做匹配。

### 4.2 从 box-only 转换

转换完全确定，不需要调用 API：

1. 读取归一化 `boxes`。
2. 每个 box 生成 `bbox_2d`。
3. 根据公式生成框中心 `point_2d`。
4. 序列化为 `solution` 字符串。
5. 使用 `datasets.Image()` 保存图像引用。

执行：

```bash
python -m tools.data.convert_to_hf_detection_dataset \
  --input /private/work/normalized/train.jsonl \
  --output-dir /private/work/hf/business_train_840
```

再与约 4k VisionReasoner 数据合并：

```bash
python -m tools.data.combine_hf_datasets \
  --input /private/work/hf/business_train_840 /private/open/visionreasoner_4k_840 \
  --output-dir /private/work/hf/stage1_business_plus_vr \
  --seed 20250715
```

### 4.3 公司 API 在 Stage 1 的用途

Stage 1 GT 不应由大模型 API 改写。API 只适合：

- 为单一业务类别生成少量等价问题表达，例如“定位所有目标”“找出图中的目标物体”。
- 如果 API 是 VLM，可在训练前做标注质检，标记“疑似漏标、框明显偏离、图像不可用”，交给人工复核。
- 生成数据审计标签或简短场景描述，但不能把 API 预测直接当 GT。

问题表达不宜过度多样化。若业务部署时始终使用固定 prompt，至少 70% Stage 1 数据应保持部署 prompt，其他表达只作为鲁棒性增强。

## 5. Stage 2：Refine SFT 数据

Stage 2 不是只给 GT box，而是给模型一组“上一次预测 proposal”，要求模型输出如何纠错。

### 5.1 推荐 20k 来源配比

按图像来源：

| 来源 | 数量 |
|---|---:|
| 业务 train | 12k |
| VisionReasoner | 4k |
| Waymo/nuImages 等自动驾驶公开数据 | 4k |

按 proposal 产生方式是另一条独立维度，推荐首版：

| Proposal 来源 | 比例 | 用途 |
|---|---:|---|
| Stage 1 模型真实预测 | 50% | 贴近当前模型的真实错误分布 |
| 程序合成错误 | 30% | 精确覆盖偏框、漏检、误检、重复框 |
| 公司 VLM API 预测 | 20% | 增加不同模型产生的错误形态 |

若公司 API 只有文本能力，第三项并入前两项，API 只负责文本增强。

### 5.2 Proposal 格式

```json
[
  {"id": "B1", "bbox_2d": [92, 126, 230, 270], "point_2d": [161, 198]},
  {"id": "B2", "bbox_2d": [600, 100, 690, 220], "point_2d": [645, 160]}
]
```

proposal 可来自 Stage 1 推理、公司 VLM API 或对 GT 的程序扰动。API 输出在进入数据集前必须解析、限制到 0–840、删除退化框并重新生成中心点。

### 5.3 从 Proposal 和 GT 计算 Refine action

不要让大模型 API 计算或修改 action。程序先用 IoU/Hungarian matching 关联 proposal 和 GT，再确定三类动作：

1. 匹配成功：输出 bbox delta 和 point delta。
2. 未匹配 proposal：输出 `delete`。
3. 未匹配 GT：加入 `new_items`。

例子：

```text
GT B1 box       = [100, 120, 220, 280]
GT B1 point     = [160, 200]
Proposal B1 box = [92, 126, 230, 270]
Proposal B1 pt  = [161, 198]

bbox delta  = GT - Proposal = [8, -6, -10, 10]
point delta = GT - Proposal = [-1, 2]
```

假设 B2 是误检，同时漏掉了 GT `[400,300,520,460]`，目标 action 为：

```json
{
  "per_bbox": [
    {
      "id": "B1",
      "action": {
        "bbox": [8, -6, -10, 10],
        "point": [-1, 2]
      }
    },
    {"id": "B2", "action": "delete"}
  ],
  "new_items": [
    {
      "bbox_2d": [400, 300, 520, 460],
      "point_2d": [460, 380]
    }
  ]
}
```

必须执行 action replay：把 action 应用到 proposal 后，最终 boxes 必须与 GT 完全一致，否则丢弃该样本。

### 5.4 ms-swift 最终 JSONL

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a precise 2D object detection assistant."
    },
    {
      "role": "user",
      "content": "<image>Locate \"target object\". Your previous predictions are [...]. Now refine them..."
    },
    {
      "role": "assistant",
      "content": "<think>B1 needs a small boundary correction, B2 is a false positive, and one object is missing.</think><answer>{\"per_bbox\":[...],\"new_items\":[...]}</answer>"
    }
  ],
  "images": ["/private/work/images_840/train/scene/frame.jpg"],
  "metadata": {
    "source_id": "sample-001",
    "scene_id": "sceneuuid",
    "scenario": "mixed"
  }
}
```

当前确定性生成器支持偏框、漏检、误检、重复框、混合错误和近正确样本：

```bash
python -m tools.data.build_refine_sft \
  --input \
    /private/work/sft/business_12k.jsonl \
    /private/work/sft/visionreasoner_4k.jsonl \
    /private/work/sft/autodrive_public_4k.jsonl \
  --output /private/work/sft/refine_sft_20k.jsonl \
  --samples 20000 \
  --seed 20250715
```

## 6. 公司大模型 API 构造 Stage 2 数据

### 6.1 API 分为两种能力

纯文本 LLM API 可以：

- 根据已经确定的 proposal、GT 和 action 生成简短、可核验的 `<think>` 理由。
- 生成少量问题表达变体。
- 对已有理由做格式修复、去重和语言统一。

它不能看图，因此不能判断 proposal 是否真的符合图像，也不能负责生成数值 GT。

多模态 VLM API 可以额外：

- 输入图像和目标名称，产生另一组 proposal，作为“待纠错的模型预测”。
- 对 Stage 1 预测做错误类型标注，例如偏框、漏检、重复框、疑似误检。
- 辅助找出可能漏标或异常图片，但只能进入人工复核队列。

无论哪种 API，GT、matching、delta、delete、new_items 和 replay 都必须由本地程序决定。

### 6.2 推荐 API 输入契约

发送给文本 API 的内容不需要包含图像，只发送脱敏结构：

```json
{
  "sample_id": "internal-hash",
  "target": "target object",
  "proposal": [
    {"id": "B1", "bbox_2d": [92, 126, 230, 270]},
    {"id": "B2", "bbox_2d": [600, 100, 690, 220]}
  ],
  "verified_actions": {
    "correct": ["B1"],
    "delete": ["B2"],
    "add_count": 1
  },
  "constraints": {
    "do_not_change_coordinates": true,
    "max_sentences": 2,
    "output_language": "English"
  }
}
```

要求 API 只返回：

```json
{
  "sample_id": "internal-hash",
  "reason": "B1 needs a small boundary correction, B2 is a false positive, and one object is missing.",
  "quality_flags": []
}
```

程序把 `reason` 放入 `<think>`，把本地已经验证过的 action JSON 原样放入 `<answer>`。API 不应返回完整 assistant answer，否则容易篡改数字。

### 6.3 推荐 API System Prompt

```text
You generate a short correction rationale for 2D object detection training.
Use only the supplied verified action summary.
Do not invent objects, coordinates, IDs, counts, or actions.
Do not repeat the full JSON.
Return strict JSON with keys sample_id, reason, and quality_flags.
The reason must be one or two concise sentences.
```

如果 API 支持 JSON Schema/structured output，应强制使用；如果不支持，则本地解析失败后最多重试两次，再退回确定性模板。

### 6.4 推荐生成比例

不建议让 API 重写全部 20k。首版可采用：

- 12k–14k：确定性模板理由，稳定且便于回归。
- 6k–8k：API 增强理由，优先覆盖 mixed、missing、duplicate 等复杂样本。
- 其中约 4k proposal 可来自公司 VLM API，但 action 仍由 GT 自动计算。

这种混合方式能增加表达和错误分布多样性，同时保留足够多完全确定、可重复的数据。

### 6.5 API 工程要求

API 构造程序至少应支持：

- endpoint、model、token 全部通过环境变量传入，绝不写入代码或 Git。
- 限并发、指数退避、超时、断点续跑。
- 用 `request_hash` 缓存，避免重复计费。
- 保存 `api_model`、`prompt_version`、`request_hash`、时间和校验结果。
- API 原始响应与最终训练 JSONL 分开保存。
- 失败样本进入重试/人工复核清单，不能静默丢失。
- 若 API 不在公司内网，未经数据安全批准不得上传业务图像。

推荐私有目录：

```text
/private/work/api_build/
  requests.jsonl
  raw_responses.jsonl
  cache/
  failed.jsonl
  validated_reasoning.jsonl
  build_manifest.json
```

当前仓库尚未实现公司 API adapter，因为还不知道接口是否兼容 OpenAI 协议、是否支持图片、鉴权方式和并发限制。拿到接口文档后，应新增独立的 `enrich_with_company_api.py`，不要把 API 调用直接塞进基础转换器；这样纯离线确定性数据仍可随时复现。

## 7. Stage 3：两轮 Refine GRPO 数据

### 7.1 默认两轮模式需要的字段

Stage 3 的输入格式与 Stage 1 基本相同：

```json
{
  "id": "hard-sample-001",
  "problem": "target object",
  "solution": "[{\"bbox_2d\":[100,120,220,280],\"point_2d\":[160,200]}]",
  "image": "<HF Image>",
  "scene_id": "train-scene-uuid"
}
```

默认 rollout 的过程是：

1. 第一轮模型根据图像自己产生 proposal。
2. interaction 将 proposal 编号为 B1、B2……并构造 refine prompt。
3. 第二轮模型输出 `per_bbox + new_items`。
4. reward 回放 action，与 `solution` GT 比较。

因此默认 Stage 3 数据不需要预存 proposal 或 SFT answer，只需要图像、目标描述和 GT。

代码也支持可选的单轮 refine 模式：数据额外提供 `proposal` 字段，直接要求模型纠错。但当前三阶段方案使用默认两轮模式。

### 7.2 8k hard-case 的构造

先用 Stage 1 或 Stage 2 模型对业务 train 推理，生成与 GT 合并的评分 JSONL：

```json
{
  "id": "sample-001",
  "scene_id": "train-scene-uuid",
  "image": "/private/path/frame.jpg",
  "boxes": [[100, 120, 220, 280]],
  "problem": "target object",
  "metrics": {"f1": 0.25, "recall": 0.5}
}
```

选择 F1 最低的 8k，同时强制 train scene allow-list：

```bash
python -m tools.data.select_hard_cases \
  --input /private/work/eval/train_proposals_with_gt.jsonl \
  --output /private/work/rl/hard_8k.jsonl \
  --samples 8000 \
  --score-key metrics.f1 \
  --hardest lowest \
  --allowed-scenes /private/work/split_v1/train_scenes.txt
```

再转为 HF Dataset：

```bash
python -m tools.data.convert_to_hf_detection_dataset \
  --input /private/work/rl/hard_8k.jsonl \
  --output-dir /private/work/hf/stage3_hard_8k
```

### 7.3 公司 API 在 Stage 3 的用途

- 不用于生成 GT 或 reward 分数。
- VLM API 可作为另一个 proposal 模型，帮助发现 Stage 1 没覆盖的错误形态。
- 文本 API 可给 hard case 打标签或生成分析报告，但这些文本默认不进入 GRPO 输入。
- hard-case 排序应以本地可复现的 Precision/Recall/F1、IoU 和格式有效率为准，而不是 API 主观评分。

## 8. 三阶段最终格式对照

| 项目 | Stage 1 Proposal GRPO | Stage 2 Refine SFT | Stage 3 Refine GRPO |
|---|---|---|---|
| 存储格式 | HF DatasetDict | JSONL | HF DatasetDict |
| 图像 | 840×840 HF Image | `images: [path]` | 840×840 HF Image |
| GT | `solution` box+中心点 | assistant refine action | `solution` box+中心点 |
| Proposal | 模型 rollout 产生 | 预先构造/推理/API 产生 | 第一轮 rollout 产生 |
| Point 来源 | box 中心自动计算 | box 中心及中心 delta | box 中心自动计算 |
| API 必需 | 否 | 否，可增强 | 否 |
| 是否允许 test | 否 | 否 | 否 |

## 9. 每批数据的强制校验

所有阶段至少检查：

1. scene 不跨 train/test，且 hard-case 全部在 `train_scenes.txt`。
2. 图像存在并可解码，尺寸为 840×840。
3. box 为整数 `xyxy`，范围在 0–840，面积大于 0。
4. `point_2d` 等于对应 box 中心。
5. Stage 2 proposal ID 唯一且按 B1、B2……编号。
6. Stage 2 action replay 后最终 box 集合与 GT 完全一致。
7. `<think>` 和 `<answer>` 标签严格闭合，answer 内只有合法 JSON。
8. API 没有修改坐标、ID、动作或目标数量。
9. 数据来源、脚本 commit、seed、API model 和 prompt version 可追溯。
10. 随机可视化检查正常后才开始 GPU smoke run。

训练前建议先分别生成 32 条样本，完成加载、prompt 展开、reward 和 action replay 的端到端检查，再扩大到 512 条和全量。
