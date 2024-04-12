export PATH=$PATH:~/.conda/envs/llava/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ichs/harrysun/anaconda3/lib/
source /wynton/protected/home/ichs/harrysun/anaconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5
conda activate llava
export CUDA_VISIBLE_DEVICES=0,1,
CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_reverse_orders.py \
    --short_eval True \
    --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/ \
    --use-qlora True \
    --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-INIT-7b-v1.5-224-lora-padding/lora_default/ \
    --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/LLaVA_heme_test_q.json \
    --image-folder /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/ \
    --answers-file answers/SFT_con_reverse.jsonl \
    --image_aspect_ratio pad \
    --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_reverse_orders.py \
    --short_eval True \
    --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/ \
    --use-qlora True \
    --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-INIT-Permuted-7b-v1.5-224-lora-padding/lora_default/ \
    --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/LLaVA_heme_test_q.json \
    --image-folder /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/ \
    --answers-file answers/SFT_con_permutated_reverse.jsonl \
    --image_aspect_ratio pad \
    --test-prompt ''
