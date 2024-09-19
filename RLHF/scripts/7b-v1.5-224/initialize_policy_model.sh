#!/bin/bash
#$ -cwd                    # Use the current working directory
#$ -j yes                   # Use the current working directory
#$ -q gpu.q
#$ -pe smp 2               # slots (threads)
#$ -l gpu_mem=35G        # Gigabytes of memory per thread (total 20 * 10G = 200 G)
#$ -R y
#$ -V
#$ -l h_rt=38:50:00        # job time

#export PATH=$PATH:~/.conda/envs/llava/bin
export PATH=$PATH:~/miniconda3/envs/LLM_env/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ibrahim/alex_schubert/miniconda3/lib
source /wynton/protected/home/ibrahim/alex_schubert/miniconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5
#conda activate llava
# source /wynton/protected/home/ichs/harrysun/anaconda3/etc/profile.d/conda.sh
# set -e
# set -x
# # module load cuda/11.5
# # conda activate llava
# # Activate the virtual environment
conda activate LLM_env
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR="/wynton/protected/group/ibrahim/alex/Dr-LLaVA/data/train_conversations.json"
export MODEL_DIR="/wynton/protected/group/ibrahim/harry/LLaVA_checkpoints" #"/wynton/protected/group/ibrahim/alex/Dr-LLaVA/experiments/MIMIC-ECG/LLaVA_checkpoints"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model

# SAVE CONFIG
MODEL_NAME=LLaVA-RL-INIT-7b-v1.5-224-lora-padding-ECG-v0

# TRAINING CONFIG
NUM_EPOCHS=4
LEARNING_RATE=1e-4
BATCH_SIZE=8
GRAD_ACCUMULATION=2

deepspeed \
    finetune_lora_sft_ds.py \
    --deepspeed scripts/zero2.json \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --image_folder "/wynton/protected/group/ibrahim/alex/Dr-LLaVA/data/image_folder" \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --query_len 1280 \
    --response_len 768 \
    --dataset "/wynton/protected/group/ibrahim/alex/Dr-LLaVA/data/train_conversations.json" \
    --dataset_format "v1" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 1000000 \
    --save_total_limit 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_aspect_ratio 'pad'
