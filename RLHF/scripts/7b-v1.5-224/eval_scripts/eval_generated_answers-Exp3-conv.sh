export PATH=$PATH:~/.conda/envs/llava/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ichs/harrysun/anaconda3/lib/
source /wynton/protected/home/ichs/harrysun/anaconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5
conda activate llava
export CUDA_VISIBLE_DEVICES=0,1,


####### Without RL #####

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_random_and_samples.py   --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KL01/checkpoint-20/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_big.json   \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp3/RL_permutated_SI_epoch20.jsonl  --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_random_and_samples.py   --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KL01/checkpoint-30/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_big.json   \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp3/RL_permutated_SI_epoch30.jsonl  --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_random_and_samples.py   --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KL01/checkpoint-40/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_big.json   \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp3/RL_permutated_SI_epoch40.jsonl  --image_aspect_ratio pad --test-prompt ''

CUDA_VISIBLE_DEVICES=$SGE_GPU python ../Eval/model_vqa_conversation_random_and_samples.py   --short_eval True    \
  --model-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF-7b-v1.5-224/sft_model/   \
  --use-qlora True --qlora-path /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RL-Fact-RLHF-7b-v1.5-224-lora-padding_KL01/checkpoint-50/adapter_model/lora_policy/  \
  --question-file /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/test_bccv_q_big.json   \
  --image-folder  /wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/ucsf_data_rl/image_folder/  \
  --answers-file results/exp3/RL_permutated_SI_epoch50.jsonl  --image_aspect_ratio pad --test-prompt ''