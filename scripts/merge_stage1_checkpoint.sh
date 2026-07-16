#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a Seg-Zero global_step_* actor checkpoint}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEGZERO_DIR="${SEGZERO_DIR:-${ROOT_DIR}/third_party/Seg-Zero}"

python3 "${SEGZERO_DIR}/training_scripts/model_merger.py" --local_dir "${CHECKPOINT_DIR%/}"
echo "Merged Hugging Face model: ${CHECKPOINT_DIR%/}/huggingface"
