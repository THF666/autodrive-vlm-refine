#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the merged Stage-1 Hugging Face model}"
: "${SFT_DATASET:?Set SFT_DATASET to refine_sft_20k.jsonl}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_PIXELS="${MAX_PIXELS:-705600}"
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES MAX_PIXELS

if swift sft --help 2>&1 | grep -q -- '--train_type'; then
  TUNER_FLAG="--train_type"
else
  TUNER_FLAG="--tuner_type"
fi

swift sft \
  --model "${MODEL_PATH%/}" \
  --dataset "${SFT_DATASET}" \
  --split_dataset_ratio 0 \
  "${TUNER_FLAG}" lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_checkpointing true \
  --deepspeed zero2 \
  --attn_impl flash_attn \
  --max_length "${MAX_LENGTH:-4096}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-16}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --output_dir "${OUTPUT_DIR%/}"
