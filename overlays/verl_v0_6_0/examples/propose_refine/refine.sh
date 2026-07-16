#!/usr/bin/env bash
set -euo pipefail

# Example RL entrypoint for the propose-refine task.
#
# Required:
#   REF_MODEL_PATH=/path/to/cold-start-or-base-model
#   SAVE_CHECKPOINT_DIR=/path/to/output/checkpoints
#
# Optional:
#   TRAIN_DATASET=Ricky06662/VisionReasoner_multi_object_7k_840
#   VAL_DATASET=$TRAIN_DATASET  # loader-only; evaluation stays disabled
#   EXPERIMENT_NAME=propose_refine_multi_object_v0_2
#   N_GPUS_PER_NODE=8

PROJECT_NAME="${PROJECT_NAME:-Propose_Refine}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-propose_refine_multi_object_v0_2}"
TRAIN_DATASET="${TRAIN_DATASET:-Ricky06662/VisionReasoner_multi_object_7k_840}"
VAL_DATASET="${VAL_DATASET:-${TRAIN_DATASET}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','tensorboard']}"

BASEDIR="${BASEDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

: "${REF_MODEL_PATH:?Set REF_MODEL_PATH to the base or cold-start checkpoint.}"
: "${SAVE_CHECKPOINT_DIR:?Set SAVE_CHECKPOINT_DIR to the checkpoint output directory.}"

export PROPOSE_REFINE_REWARD_DEBUG="${PROPOSE_REFINE_REWARD_DEBUG:-1}"
export PROPOSE_REFINE_INTERACTION_DEBUG="${PROPOSE_REFINE_INTERACTION_DEBUG:-0}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="${BASEDIR}/examples/propose_refine/configs" \
    --config-name='propose_refine_grpo' \
    data.train_files="${TRAIN_DATASET}" \
    data.val_files="${VAL_DATASET}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=1536 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.image_key=image \
    data.prompt_key=problem \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH%/}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1.0e-2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${BASEDIR}/examples/propose_refine/configs/propose_refine_interaction.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="ignore_strippable" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    +trainer.max_global_ckpt_to_keep=1 \
    trainer.test_freq=-1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    +trainer.tensorboard_dir="${SAVE_CHECKPOINT_DIR}/logs/tensorboard" \
    +trainer.rl_logging_board_dir="${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
