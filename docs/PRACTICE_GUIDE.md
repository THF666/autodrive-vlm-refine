# 三阶段实践指南

以下路径都是示例。所有 `data/`、`work/`、模型、日志和 split 输出都应位于公司服务器私有磁盘，不要放入 Git 仓库。

## 0. 先建立不可泄漏的 90/10 切分

```bash
python -m tools.data.build_scene_split \
  --scene-stats /private/raw/scene_stats.txt \
  --annotation-dir /private/raw/annotations \
  --output-dir /private/work/split_v1 \
  --test-ratio 0.10 \
  --seed 20250715
```

对当前数据，目标是：

- 总计：2391 scenes / 57437 images。
- Test：固定 239 scenes，图像数尽量接近 5744。
- Train：固定 2152 scenes，图像数约 51693。

先检查 `/private/work/split_v1/split_report.txt`。实际 test 图像数可能与 5744 相差少量，这是“完整场景不拆分”的正常结果。

分别归一化 train/test；下面假定原始 JSON 使用 `image` 和 `bbox: xyxy`：

```bash
python -m tools.data.normalize_annotations \
  --annotation-dir /private/raw/annotations \
  --annotation-manifest /private/work/split_v1/train_annotations.txt \
  --image-root /private/raw/images \
  --prepared-image-dir /private/work/images_840/train \
  --output-jsonl /private/work/normalized/train.jsonl \
  --problem 'target object'

python -m tools.data.normalize_annotations \
  --annotation-dir /private/raw/annotations \
  --annotation-manifest /private/work/split_v1/test_annotations.txt \
  --image-root /private/raw/images \
  --prepared-image-dir /private/work/images_840/test \
  --output-jsonl /private/work/normalized/test.jsonl \
  --problem 'target object'
```

第一轮只用 20 张图跑通，人工打开生成图检查 box 缩放，再处理全部数据。若 JSON schema 不同，按 [数据契约](DATA_CONTRACT.md) 指定 key。

## 1. Stage 1：Seg-Zero Proposal GRPO

把全部业务 train 转成 Seg-Zero/veRL 的 HF save-to-disk 格式：

```bash
python -m tools.data.convert_to_hf_detection_dataset \
  --input /private/work/normalized/train.jsonl \
  --output-dir /private/work/hf/business_train_840
```

将 VisionReasoner 开源 4k 与业务数据合并。开源数据需具有 `id/problem/solution/image` 四列：

```bash
python -m tools.data.combine_hf_datasets \
  --input /private/work/hf/business_train_840 /private/open/visionreasoner_4k_840 \
  --output-dir /private/work/hf/stage1_business_plus_vr \
  --seed 20250715
```

先做三级闸门：32 条、512 条、全量。32/512 数据可用 HF Dataset 的 `select()` 单独保存。只有以下条件都成立才开全量：reward 非恒定、有效 JSON 比例上升、框不集中到边界、checkpoint 能合并并推理。

全量命令：

```bash
export MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct
export TRAIN_DATASET=/private/work/hf/stage1_business_plus_vr
export SAVE_CHECKPOINT_DIR=/private/checkpoints
export EXPERIMENT_NAME=stage1_business_proposal_grpo
bash scripts/run_stage1_segzero.sh
```

合并选定 checkpoint：

```bash
export CHECKPOINT_DIR=/private/checkpoints/stage1_business_proposal_grpo/global_step_XXX/actor
bash scripts/merge_stage1_checkpoint.sh
```

注意：当前还没有业务效果证据。不要仅凭 loss/reward 曲线选模型；至少在固定 test 上记录 format-valid rate、Precision/Recall/F1、AP50、AP75 和 COCO mAP。

## 2. Stage 2：20k Refine 冷启动 SFT

推荐首轮配比：业务 train 12k + VisionReasoner 4k + Waymo/nuImages 4k。业务 12k 用场景轮询抽样，避免长视频场景垄断：

```bash
python -m tools.data.sample_by_scene \
  --input /private/work/normalized/train.jsonl \
  --output /private/work/sft/business_12k.jsonl \
  --samples 12000 \
  --seed 20250715
```

公开数据也转换为同一归一化 JSONL 后构造 SFT：

```bash
python -m tools.data.build_refine_sft \
  --input \
    /private/work/sft/business_12k.jsonl \
    /private/work/sft/visionreasoner_4k.jsonl \
    /private/work/sft/autodrive_public_4k.jsonl \
  --output /private/work/sft/refine_sft_20k_draft.jsonl \
  --samples 20000 \
  --seed 20250715
```

数据生成器的默认错误分布为：偏框 32%、漏检 18%、误检 14%、重复框 10%、混合错误 20%、近正确 6%。首轮保持不变；待业务错误统计出来后再按真实分布调权重。这个文件只是 oracle answer 已验证的 draft，其中 `<think>` 是占位模板，不能直接作为最终 CoT 冷启动数据。

导出公司 VLM CoT 请求：

```bash
python -m tools.data.prepare_refine_cot_requests \
  --input /private/work/sft/refine_sft_20k_draft.jsonl \
  --output /private/work/api_build/cot_requests.jsonl
```

公司 API 调用程序读取每行的 `image + messages`，让 VLM 根据图像、query、proposal 和锁定的 oracle action 只返回 `request_id + cot`。得到 `/private/work/api_build/cot_responses.jsonl` 后合并：

```bash
python -m tools.data.merge_refine_cot_responses \
  --draft /private/work/sft/refine_sft_20k_draft.jsonl \
  --responses /private/work/api_build/cot_responses.jsonl \
  --output /private/work/sft/refine_sft_20k_vlm_cot.jsonl
```

合并器只替换 `<think>`，不会采用 API 返回的任何 box/action。详细请求、响应和质检契约见 [三阶段数据准备手册](THREE_STAGE_DATA_PREPARATION.md)。

运行 LoRA SFT：

```bash
export MODEL_PATH=/private/checkpoints/stage1/.../actor/huggingface
export SFT_DATASET=/private/work/sft/refine_sft_20k_vlm_cot.jsonl
export OUTPUT_DIR=/private/checkpoints/stage2_refine_sft
bash scripts/run_stage2_swift_sft.sh
```

训练后先用 100–300 条训练 probe 检查严格格式与 action replay，再对 test 做一次阶段性评测。选定 checkpoint 后合并 LoRA：

```bash
export ADAPTER_PATH=/private/checkpoints/stage2_refine_sft/checkpoint-XXX
export MERGED_MODEL_PATH=/private/checkpoints/stage2_refine_sft_merged
bash scripts/merge_stage2_lora.sh
```

## 3. Stage 3：8k hard-case Refine GRPO

用 Stage 1/2 模型对业务 **train** 推理并与 GT 计算指标。选 F1 最低的 8k：

```bash
python -m tools.data.select_hard_cases \
  --input /private/work/eval/train_proposals_with_gt.jsonl \
  --output /private/work/rl/hard_8k.jsonl \
  --samples 8000 \
  --score-key metrics.f1 \
  --hardest lowest \
  --allowed-scenes /private/work/split_v1/train_scenes.txt
```

转为 Stage 3 HF 数据：

```bash
python -m tools.data.convert_to_hf_detection_dataset \
  --input /private/work/rl/hard_8k.jsonl \
  --output-dir /private/work/hf/stage3_hard_8k
```

启动两轮 GRPO：

```bash
export REF_MODEL_PATH=/private/checkpoints/stage2_refine_sft_merged
export TRAIN_DATASET=/private/work/hf/stage3_hard_8k
export SAVE_CHECKPOINT_DIR=/private/checkpoints
export EXPERIMENT_NAME=stage3_business_refine_grpo_8k
bash scripts/run_stage3_refine_grpo.sh
```

默认 rollout 中第一轮产生 proposal，第二轮输出 refine action；overlay 会对 proposal turn 做 loss mask，让 PPO 主要优化 refine turn，同时在 proposal 解析失败时避免全零 token mask。

veRL v0.6.0 会无条件实例化 `val_dataset`，所以脚本用 train 路径满足 loader，但同时设置 `val_before_train=false` 和 `test_freq=-1`，不会执行验证。这不是额外划分的验证集。

## 4. 不使用验证集时的纪律

- Test 只能在：基础模型、Stage 1 完成、Stage 2 完成、Stage 3 完成时各评一次。
- 不根据 test 结果频繁改超参、选 step 或反复早停，否则 test 实际上变成验证集。
- 训练中仅看 reward、KL、格式率、梯度、显存和少量 train probe。
- 所有阶段使用同一份 `split_manifest.json` 和 seed，禁止重新抽 test。
