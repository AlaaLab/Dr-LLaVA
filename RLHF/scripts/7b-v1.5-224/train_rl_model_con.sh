#!/bin/bash

export PATH=$PATH:~/.conda/envs/llava/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ichs/harrysun/anaconda3/lib/
source /wynton/protected/home/ichs/harrysun/anaconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5
conda activate llava
export CUDA_VISIBLE_DEVICES=0,1,
export DATA_DIR="/path/to/your/data/directory"
export MODEL_DIR="/wynton/protected/project/outcome_pred/LLaVA_checkpoints/"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1

# MODEL CONFIG
POLICY_BASE_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model
RM_BASE_MODEL_NAME=LLaVA-RLHF-13b-v1.5-336/sft_model

POLICY_LORA=LLaVA-RL-INIT-7b-v1.5-224-lora-padding/lora_default
RM_LORA=LLaVA-Fact-RM-13b-v1.5-336-lora-padding/checkpoint-200  # we use early stopping

# SAVE CONFIG
MODEL_NAME=LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding-conv

# TRAINING CONFIG
LEARNING_RATE=3e-5
KL_COEF=0.1
EPOCH=4
ROLLOUT_BATCH_SIZE=512 
STEP_BATCH_SZIE=256
ROLLOUT_PER_DEVICE_BATCH_SIZE=16
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=8
STEP_PER_DEVICE_BATCH_SIZE=8
NOPTEPOCHS=2
# FACT-RLHF CONFIG
INCOMPLETE_RESPONSE=-2.0 # was -8 before
LENGTH_BONUS=-10.0
CORRECT_BONUS=2.0

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_ppo.py \
    --do_train \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --base_model_name "$MODEL_DIR/$POLICY_BASE_MODEL_NAME" \
    --reward_base_model_name "$MODEL_DIR/$RM_BASE_MODEL_NAME" \
    --policy_model_name_or_path "$MODEL_DIR/$POLICY_LORA" \
    --reward_model_name_or_path "$MODEL_DIR/$RM_LORA" \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --warmup_steps 5 \
    --dataset_path "/wynton/protected/project/outcome_pred/LLaVA_checkpoints/ucsf_data_rl/LLaVA_heme_train_rl.json" \
    --train_splits "train" \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --penalty_reward_value $INCOMPLETE_RESPONSE \
    --length_bonus_score $LENGTH_BONUS \
    --correct_bonus_score $CORRECT_BONUS \
    --relative_stop_token_penalty True \
    --penalize_no_stop_token True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --temperature 1.0 \
    --whiten_rewards True \
    --model_max_length 2048 \
    --query_len 256 \
    --response_len 224 \
    --noptepochs $NOPTEPOCHS \
    --image_folder "/wynton/protected/project/outcome_pred/LLaVA_checkpoints/ucsf_data_rl/image_folder" \
    --vision_tower different \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --reward_prompt_file "./prompts/fact_rlhf_reward_prompt.txt" \
    --image_aspect_ratio 'pad'
