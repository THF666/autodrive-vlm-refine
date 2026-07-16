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

### 4.3 公司 VLM API 不参与 Stage 1 box 构造

公司 VLM API 的定位是为 Stage 2 构造图像条件 CoT，不用于预测 Stage 1 的 box。Stage 1 的 GT 只来自人工 box 标注，proposal 由训练中的 Qwen2.5-VL rollout 产生。

若业务部署时始终使用固定 prompt，Stage 1 应直接使用真实类别名称和部署 prompt，例如 `vehicle` 或具体业务目标名称，不能用含义模糊的 `target object` 代替真实类别。

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
| Stage 1 模型真实预测 | 60% | 贴近当前 Qwen2.5-VL 的真实错误分布 |
| 程序合成错误 | 40% | 精确覆盖偏框、漏检、误检、重复框 |

公司 VLM API 不负责产生 proposal。它在 proposal 和 oracle refine action 都确定后查看图像，为 SFT assistant answer 生成视觉 CoT。

### 5.2 Proposal 格式

```json
[
  {"id": "B1", "bbox_2d": [92, 126, 230, 270], "point_2d": [161, 198]},
  {"id": "B2", "bbox_2d": [600, 100, 690, 220], "point_2d": [645, 160]}
]
```

proposal 只来自 Stage 1 推理或对 GT 的程序扰动。所有 proposal 在进入数据集前都必须解析、限制到 0–840、删除退化框并重新生成中心点。

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

## 6. 使用公司 VLM API 构造 Stage 2 CoT

### 6.1 与原 `verl_fixed_2turns_v2` 数据对齐

原项目的冷启动文件 `grefcoco_1turn_refine_ms_swift_sft_840_saved_images_xyxy.json` 有 5150 条样本。每条数据都是：

```text
输入：图像 + 目标 query + previous predictions + Refine 输出规范
输出：<think>图像条件的视觉纠错过程</think>
     <answer>per_bbox/delete/new_items</answer>
```

其中 `<think>` 不是通用文本模板，而是结合图像内容判断候选框是否匹配目标、是否覆盖完整、是否为误检以及是否存在漏检。公司 VLM API 应当复现的正是这个“视觉 CoT 教师”角色，不是检测器角色。

### 6.2 正确的数据构造顺序

```text
人工 GT box
  → Stage 1 推理或程序扰动得到 proposal
  → 程序匹配 proposal 与 GT
  → 程序计算并 replay 验证 oracle refine action
  → 图像 + query + proposal + 已锁定 action 发送给公司 VLM
  → VLM 只生成视觉 CoT
  → 程序把 CoT 放入 <think>，把原 oracle action 放入 <answer>
  → 最终格式校验和抽样人工检查
```

这样 VLM 可以根据图像写出有内容的推理，同时不会产生错误 delta、遗漏 `new_items` 或篡改 GT。

### 6.3 VLM 请求内容

每个请求必须包含：

- 840×840 图像。
- 使用真实业务类别名称的 query。
- 带 B1、B2……ID 的 previous predictions。
- 已经由程序计算并 replay 通过的 oracle refine answer。
- 简化 action summary：哪些 ID 调整、哪些删除、漏了几个目标。

VLM 得到 oracle answer 是为了生成与正确动作一致的视觉解释；它不需要也不允许重新计算 answer。

仓库可先导出与具体 API 协议无关的请求 JSONL：

```bash
python -m tools.data.prepare_refine_cot_requests \
  --input /private/work/sft/refine_sft_20k_draft.jsonl \
  --output /private/work/api_build/cot_requests.jsonl
```

每行请求结构：

```json
{
  "request_id": "sha256...",
  "image": "/private/work/images_840/train/scene/frame.jpg",
  "messages": [
    {"role": "system", "content": "Inspect the image and generate visual correction reasoning..."},
    {"role": "user", "content": "query + proposal + VERIFIED_REFINE_ACTION + ACTION_SUMMARY"}
  ],
  "oracle_answer": {
    "per_bbox": [],
    "new_items": []
  },
  "action_summary": {
    "correct_or_adjust": ["B1"],
    "delete": ["B2"],
    "add_count": 1
  }
}
```

具体 API adapter 读取 `image + messages`，按照公司协议上传图片并调用 VLM。

### 6.4 VLM 输出契约

要求公司 VLM 严格返回：

```json
{
  "request_id": "sha256...",
  "cot": "B1 visually corresponds to the target but its boundary is too loose on the right. B2 covers a background distractor, and another target instance is visible farther ahead but was missed.",
  "model": "company-vlm-model-name"
}
```

CoT 应满足：

- 必须基于图像描述目标类别、可见属性、相对位置或候选框覆盖情况。
- 应解释需要 correction、delete 或 add 的视觉原因。
- 不输出 `<think>`、`<answer>` 标签。
- 不重复 JSON answer，不修改坐标、ID、动作和数量。
- 不出现“根据 GT”“oracle 告诉我”等泄漏教师信息的表述。
- 建议 1–4 句；多目标复杂样本可以稍长，但不写空泛的通用模板。

### 6.5 合并 CoT 并锁定 answer

API 原始响应保存为 JSONL 后执行：

```bash
python -m tools.data.merge_refine_cot_responses \
  --draft /private/work/sft/refine_sft_20k_draft.jsonl \
  --responses /private/work/api_build/cot_responses.jsonl \
  --output /private/work/sft/refine_sft_20k_vlm_cot.jsonl
```

合并器会：

1. 用确定性 request ID 对齐请求和响应。
2. 拒绝缺失、重复、过短、过长或带 `<answer>` 标签的 CoT。
3. 只替换 `<think>` 内容。
4. 保留原始已验证 `<answer>`，不采用 VLM 返回的任何数值答案。
5. 记录 `cot_source=company_vlm_api`、request ID 和 API model。

工具同时支持本项目的 `messages + images` JSONL，以及原 Refine 项目的 `conversations + <img>path</img>` JSON list。

### 6.6 20k CoT 生成规模与质检

目标是得到 20k 条通过校验的 VLM CoT，而不是 20k 个 API 原始响应。建议先发 23k–25k 个候选请求，经过以下过滤后保留 20k：

- JSON/标签/长度不合法。
- CoT 与 action 冲突，例如 oracle 是 delete，但理由声称候选正确且应保留。
- CoT 描述的类别与业务目标不一致。
- 多条样本出现高度重复的套话。
- 图像不可用、proposal/GT/action replay 失败。

先人工检查至少 300 条，按 `jitter/missing/false_positive/duplicate/mixed/near_correct` 分层抽样。重点检查 CoT 是否真的看图、是否与 locked answer 一致，而不只是语言是否通顺。

### 6.7 API 工程要求

- endpoint、model、token 全部通过环境变量传入，绝不写入代码或 Git。
- 限并发、指数退避、超时、断点续跑。
- 用 `request_id` 缓存，避免重复计费。
- 保存 API model、prompt version、时间、原始响应和校验结果。
- 失败样本进入重试/人工复核清单，不能静默丢失。
- 若 API 不在公司内网，未经数据安全批准不得上传业务图像。

当前已经实现请求导出和响应合并；直接调用公司 API 的 adapter 仍需接口协议，包括 endpoint、鉴权、图片传输方式、响应字段和并发限制。

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

### 7.3 公司 VLM API 不参与 Stage 3 rollout

公司 VLM API 只用于 Stage 2 冷启动 CoT 构造，不用于 Stage 3 的 proposal、GT、reward 或 hard-case 评分。Stage 3 的 proposal 必须由正在训练的 Qwen2.5-VL 第一轮 rollout 产生，hard-case 排序使用本地可复现的 Precision/Recall/F1、IoU 和格式有效率。

## 8. 三阶段最终格式对照

| 项目 | Stage 1 Proposal GRPO | Stage 2 Refine SFT | Stage 3 Refine GRPO |
|---|---|---|---|
| 存储格式 | HF DatasetDict | JSONL | HF DatasetDict |
| 图像 | 840×840 HF Image | `images: [path]` | 840×840 HF Image |
| GT | `solution` box+中心点 | assistant refine action | `solution` box+中心点 |
| Proposal | 模型 rollout 产生 | 预先构造/推理/API 产生 | 第一轮 rollout 产生 |
| Point 来源 | box 中心自动计算 | box 中心及中心 delta | box 中心自动计算 |
| 公司 VLM API | 不使用 | 用于构造图像条件 CoT | 不使用 |
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
