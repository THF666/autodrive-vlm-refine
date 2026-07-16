# Autodrive VLM Refine

面向自动驾驶前视图像、单类 2D box 检测的 Qwen2.5-VL 三阶段训练工作区。

当前基线固定为 `Qwen2.5-VL-7B-Instruct`：Stage 1 使用 Seg-Zero/VisionReasoner Proposal GRPO，Stage 2 使用 ms-swift 做 Refine 冷启动 SFT，Stage 3 使用 veRL v0.6.0 做两轮 Propose→Refine GRPO。

## 当前结论

- 2391 个场景、57437 张图像必须按场景切分。脚本固定选 239 个完整场景作为测试集，并把测试图像数优化到 5744 附近；其余约 51693 张是训练集。
- 测试集只用于阶段性最终评测，绝不用于 SFT、RL、hard-case 选择或 checkpoint 挑选。项目不设验证集。
- 约 5.17 万张业务训练图全部进入 Stage 1；Stage 2 的 20k 和 Stage 3 的 8k 是专项子集，不代表其余业务数据被丢弃。
- 现状是“Stage 1 在开源数据上跑通”，不是“Stage 1 已训练有效”。必须先做业务数据 smoke run 和固定测试集基线，再开始长训。

## 数据配比

| 阶段 | 数据 | 建议规模 | 目的 |
|---|---:|---:|---|
| Stage 1 Proposal GRPO | 全部业务 train + VisionReasoner 开源数据 | 约 51.7k + 4k | 业务域迁移、多目标 box/point 输出 |
| Stage 2 Refine SFT | 业务 12k + VisionReasoner 4k + 自动驾驶公开数据 4k | 20k | 公司 VLM 构造视觉 CoT，程序锁定 refine action |
| Stage 3 Refine GRPO | Stage 1/2 在业务 train 上的最难样本 | 8k | 用奖励继续强化实际纠错能力 |
| Test | 完整隔离的约 5.7k 业务图像 | 10% | 只做最终比较 |

公开自动驾驶 2D 数据优先级：Waymo Perception 提供原生 camera 2D boxes；nuScenes 主体是 3D 标注，若需要原生 2D boxes，优先使用同系列的 nuImages。参见 [Waymo Perception](https://waymo.com/open/data/perception/) 与 [nuImages](https://www.nuscenes.org/nuimages)。如果公开数据转换暂时阻塞，首轮可先用“业务 16k + VisionReasoner 4k”，不要把测试集补进来。

## 仓库结构

```text
configs/                 只含无敏感信息的环境变量模板
docs/                    数据契约、实践指南、服务器操作手册
overlays/verl_v0_6_0/    从现有 verl_fixed_2turns_v2 提取的最小两轮 RL overlay
scripts/                 三阶段训练、模型合并、上游初始化脚本
tools/data/              场景切分、标注归一化、SFT 构造、hard-case 选择
tests/                   不含业务 UUID 的合成单元测试
```

`third_party/`、`data/`、`outputs/`、`checkpoints/`、日志和权重均被 `.gitignore` 排除。公开仓库不能提交业务 UUID、标注清单、图片、路径、模型或训练日志。

## 最短启动路径

公司服务器上的 Git Bash/Linux shell：

```bash
git clone https://github.com/THF666/autodrive-vlm-refine.git
cd autodrive-vlm-refine
bash scripts/bootstrap_upstreams.sh
python -m unittest discover -s tests -v
```

之后严格按 [三阶段实践指南](docs/PRACTICE_GUIDE.md) 执行。三个阶段的精确格式、box-only 转换和公司大模型 API 构造方式见 [三阶段数据准备手册](docs/THREE_STAGE_DATA_PREPARATION.md)；标注 JSON 字段不确定时看 [数据契约](docs/DATA_CONTRACT.md)，安装和同步见 [服务器手册](docs/SERVER_RUNBOOK.md)。

## 上游边界与可行性

- Seg-Zero 固定到 commit `5507720`，其官方代码明确支持 Qwen2-VL/Qwen2.5-VL。
- Stage 3 固定到 veRL release `v0.6.0`（commit `ddd86f527a4af75095e4677b02b5aa272913a088`）。当前 overlay 已对该 tag 做过目录、基类 API 和 Python 编译检查。
- Stage 2 使用 ms-swift 标准 `messages + images` JSONL；官方文档支持本地 JSONL 多模态 SFT。训练脚本会自动兼容 `--train_type`/`--tuner_type` 参数名差异。
- Qwen3-VL 不是简单换模型路径：Seg-Zero 的模型注册、Transformers/vLLM 版本，Stage 3 的 Qwen2-VL RoPE 逻辑和 SGLang/veRL 版本都要改。当前先使用 Qwen2.5-VL 是风险最低的方案。

## 本地同步方式

你仍然只需要 Git Bash：

```bash
cd /c/Study/HW/CoL-infer/autodrive-vlm-refine
git switch agent/bootstrap-training-workspace
git pull
```

功能合并到 `main` 后，公司服务器只执行 `git switch main && git pull`。不需要在公司服务器使用 AI，也不需要把业务数据推到 GitHub。

## License

Apache-2.0。第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
