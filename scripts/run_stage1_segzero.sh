#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-VL-7B-Instruct}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to the combined HF save_to_disk dataset}"
: "${SAVE_CHECKPOINT_DIR:?Set SAVE_CHECKPOINT_DIR}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEGZERO_DIR="${SEGZERO_DIR:-${ROOT_DIR}/third_party/Seg-Zero}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage1_business_proposal_grpo}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"

cd "${SEGZERO_DIR}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main \
  config=training_scripts/visionreasoner_7b.yaml \
  data.train_files="${TRAIN_DATASET}" \
  data.val_files=None \
  data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
  worker.actor.model.model_path="${MODEL_PATH%/}" \
  worker.actor.kl_loss_coef=1.0e-2 \
  worker.actor.optim.lr=1.0e-6 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.rollout.tensor_parallel_size=2 \
  worker.rollout.gpu_memory_utilization=0.55 \
  worker.rollout.enable_chunked_prefill=false \
  worker.rollout.n="${ROLLOUT_N}" \
  worker.reward.compute_score=vision_reasoner \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.total_episodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.val_before_train=false \
  trainer.save_checkpoint_path="${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
