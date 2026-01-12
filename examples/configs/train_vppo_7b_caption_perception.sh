#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

# 为分阶段训练生成统一的 WandB Run ID，使两个 epoch 的曲线连续
if [ -z "$WANDB_RUN_ID" ]; then
    export WANDB_RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
fi
export WANDB_RESUME="allow"

export WANDB_API_KEY=2639319e5c431af3ea5a25aa004b985328a8067c

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

MODEL_PATH="/gemini/space/telemem/model_zoo/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
rollout=8

top_p_perception_tokens=0.4
advantage_scaling_min=0.9
entropy_penalty_coef=0.06

# EXP_NAME for this script
EXP_NAME="vppo_7b_caption0.02_then_perception0.01_pen0.06_klreward"
SAVE_PATH="./checkpoints/${EXP_NAME}"

CONGI_FILE="examples/configs/config.yaml"
TRAIN_FILE="/gemini/space/telemem/ljx/VPPO_ViRL39K_train/*_captioned.parquet"
VAL_FILE="/gemini/space/telemem/ljx/VPPO_MMK12_validation"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"

# Epoch 1: Only use caption kl loss
echo "Starting Epoch 1: Caption KL loss only"
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=1 \
    trainer.save_freq=1 \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.project_name="7b_vppo" \
    trainer.logger=['console','wandb'] \
    algorithm.use_vppo_on_entropy=False \
    algorithm.use_vppo_on_perception=False \
    algorithm.use_perception_kl_reward=False \
    algorithm.perception_kl_coef=0.01 \
    algorithm.use_caption_kl_reward=True \
    algorithm.caption_kl_coef=0.02 \
    algorithm.use_advantage_shaping=False \
    algorithm.use_entropy_penalty=True \
    algorithm.top_p_perception_tokens=${top_p_perception_tokens} \
    algorithm.entropy_penalty_coef=${entropy_penalty_coef} \
    algorithm.advantage_scaling_min=${advantage_scaling_min} \
    worker.rollout.n=${rollout} \
    worker.rollout.limit_images=8 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.micro_batch_size_per_device_for_update=16

# Epoch 2: Only use perception kl loss
echo "Starting Epoch 2: Perception KL loss only"
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=2 \
    trainer.find_last_checkpoint=True \
    trainer.save_freq=1 \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.project_name="7b_vppo" \
    trainer.logger=['console','wandb'] \
    algorithm.use_vppo_on_entropy=False \
    algorithm.use_vppo_on_perception=False \
    algorithm.use_perception_kl_reward=True \
    algorithm.perception_kl_coef=0.01 \
    algorithm.use_caption_kl_reward=False \
    algorithm.caption_kl_coef=0.02 \
    algorithm.use_advantage_shaping=False \
    algorithm.use_entropy_penalty=True \
    algorithm.top_p_perception_tokens=${top_p_perception_tokens} \
    algorithm.entropy_penalty_coef=${entropy_penalty_coef} \
    algorithm.advantage_scaling_min=${advantage_scaling_min} \
    worker.rollout.n=${rollout} \
    worker.rollout.limit_images=8 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.micro_batch_size_per_device_for_update=16

