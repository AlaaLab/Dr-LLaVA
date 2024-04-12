export PATH=$PATH:~/.conda/envs/llava/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ichs/harrysun/anaconda3/lib/
source /wynton/protected/home/ichs/harrysun/anaconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5
conda activate llava
export CUDA_VISIBLE_DEVICES=0,1,

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py    --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-INIT-7b-v1.5-224-lora-padding/lora_default/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json  \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_RQ-R.jsonl --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py    --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-INIT-7b-v1.5-224-lora-padding-permutate/lora_default/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json  \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_permutated_RQ-R.jsonl --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py   --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KLsmall/checkpoint-10/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_permutated_RQ-R_v0.jsonl  --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py    --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KL01/checkpoint-10/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json  \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_permutated_RQ-R_v1.jsonl --image_aspect_ratio pad --test-prompt ''


CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py  --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_bias_alignment/checkpoint-10/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json  \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_permutated_RQ-R_v2.jsonl --image_aspect_ratio pad --test-prompt ''


CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa.py  --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_bias_alignment_Jan_27/checkpoint-10/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_hint_cor.json  \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp2/SFT_permutated_RQ-R_v3.jsonl --image_aspect_ratio pad --test-prompt ''