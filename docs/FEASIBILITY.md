# 可行性判断与 Qwen3-VL 迁移边界

## Qwen2.5-VL

当前方案可行，且是最小改动路径：

1. Seg-Zero 当前版本直接注册了 Qwen2.5-VL，训练脚本也以 `Qwen2.5-VL-7B-Instruct` 为基座。
2. ms-swift 原生接受本地多模态 `messages + images` JSONL，适合只做一次冷启动 SFT。
3. `verl_fixed_2turns_v2` 的核心基线文件与官方 veRL v0.6.0 一致；本仓库只叠加 dataset、reward、interaction 和 agent-loop mask。
4. Stage 1 和 Stage 3 的输出契约都使用 0–840 的 `bbox_2d + point_2d`，Stage 2 纠错 action 与 Stage 3 reward 能对齐。

尚未验证的是训练效果和完整 8×A100 运行，不是代码入口：当前只有 Stage 1 开源数据 smoke-run 经验。实践上必须从 32 条数据开始逐级放大。

## 为什么暂不切 Qwen3-VL

Qwen3-VL 至少涉及以下代码与依赖迁移：

- Seg-Zero 的模型 registry、processor 与 monkey patch。
- Transformers、vLLM 与 Qwen VL utils 版本联动。
- `CustomRLHFDataset` 中对 `Qwen2VLImageProcessor` 的判断和 `verl.models.transformers.qwen2_vl.get_rope_index`。
- veRL/SGLang 的 Qwen3-VL 多模态 rollout 支持与 tokenization sanity check。
- bbox 坐标模板、chat template 和模型输出格式回归测试。

因此“只改 MODEL_PATH”不可行。等 Qwen2.5 三阶段闭环和评测脚本稳定后，再单独开 Qwen3-VL 迁移分支，先完成推理与 dataset encode 单测，再做 RL smoke-run。
