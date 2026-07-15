#!/usr/bin/env bash
set -euo pipefail

: "${ADAPTER_PATH:?Set ADAPTER_PATH to the selected ms-swift checkpoint}"
: "${MERGED_MODEL_PATH:?Set MERGED_MODEL_PATH for the Stage-3 full model}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift export \
  --adapters "${ADAPTER_PATH%/}" \
  --merge_lora true \
  --output_dir "${MERGED_MODEL_PATH%/}"
