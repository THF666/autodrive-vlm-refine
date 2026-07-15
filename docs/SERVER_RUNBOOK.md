# 公司服务器操作手册

## 1. Git Bash 工作流

第一次：

```bash
git clone https://github.com/THF666/autodrive-vlm-refine.git
cd autodrive-vlm-refine
git switch agent/bootstrap-training-workspace
git pull
```

功能 PR 合并后改回稳定分支：

```bash
git switch main
git pull
```

以后每次同步只需：

```bash
cd /path/to/autodrive-vlm-refine
git pull
```

业务文件永远放在仓库外，例如 `/data/company/...` 与 `/work/company/...`。

## 2. 初始化上游代码

```bash
bash scripts/bootstrap_upstreams.sh
```

它会得到：

- `third_party/Seg-Zero` at commit `5507720`。
- `third_party/verl` at tag `v0.6.0`，并自动覆盖本项目的 Propose/Refine 文件。

不要手工在 `third_party` 内长期改代码；要同步的修改放回本仓库 `overlays/`。

## 3. 三套独立环境

三个框架依赖的 Transformers、vLLM/SGLang 版本不同，不要强行装进一个 conda 环境。

Stage 1 按 Seg-Zero 上游要求：

```bash
conda create -n stage1-segzero python=3.12 -y
conda activate stage1-segzero
pip install torch==2.6.0 torchvision==0.21.0
pip install -e third_party/Seg-Zero
pip install -e '.[segzero]'
```

Stage 2 单独安装 ms-swift，并在首次成功运行后记录精确版本：

```bash
conda create -n stage2-swift python=3.11 -y
conda activate stage2-swift
pip install -U ms-swift
```

Stage 3 按 veRL v0.6.0 的 SGLang extra：

```bash
conda create -n stage3-verl python=3.11 -y
conda activate stage3-verl
pip install uv
cd third_party/verl
python -m uv pip install -e '.[sglang]'
cd ../..
```

具体 CUDA/PyTorch 组合以公司镜像和驱动为准。每个阶段首次成功后运行：

```bash
bash scripts/record_environment.sh /private/run_metadata/stageX_first_success
```

该目录可能暴露内部环境，只保存在服务器，不提交 GitHub。

## 4. 报错回传

优先复制以下最小信息到本地：

- 完整命令和环境变量名，但删掉 token、公司绝对路径和 UUID。
- traceback 前后各 50 行。
- `pip freeze` 中相关包版本。
- GPU 型号、CUDA、PyTorch 版本。
- 一条脱敏后的输入 JSON；不要发真实图像或业务 ID。

本地修复后由这里提交 GitHub，公司服务器只执行 `git pull` 和重跑。
