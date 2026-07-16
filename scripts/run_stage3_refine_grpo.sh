#!/usr/bin/env bash
set -euo pipefail

: "${REF_MODEL_PATH:?Set REF_MODEL_PATH to the merged Stage-2 model}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to the hard-case HF dataset}"
: "${SAVE_CHECKPOINT_DIR:?Set SAVE_CHECKPOINT_DIR}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_DIR="${VERL_DIR:-${ROOT_DIR}/third_party/verl}"

cd "${VERL_DIR}"
export REF_MODEL_PATH TRAIN_DATASET SAVE_CHECKPOINT_DIR
# veRL v0.6.0 always instantiates a val dataset. Reuse train only to satisfy
# the loader; refine.sh sets val_before_train=false and test_freq=-1.
export VAL_DATASET="${VAL_DATASET:-${TRAIN_DATASET}}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage3_business_refine_grpo_8k}"
export LOG_DIR="${LOG_DIR:-${SAVE_CHECKPOINT_DIR}/logs}"

bash examples/propose_refine/refine.sh
