#!/bin/bash
#$ -S /bin/bash            # the shell language when run via the job scheduler
#$ -N test_proc_3_agg      # Job name
#$ -cwd                    # Use the current working directory
#$ -pe smp 2               # slots (threads)
#$ -l mem_free=20G        # Gigabytes of memory per thread (total 20 * 10G = 200 G)
#$ -l scratch=100G         # GiB of /scratch space
#$ -l h_rt=20:00:00        # job time

# Generate a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Activate the virtual environment
conda activate LLM_env
# source /wynton/protected/home/ibrahim/alex_schubert/miniconda3/envs/LLM_env/bin/activate

# Define log file with timestamp
LOG_FILE="ecg_processing_job_${TIMESTAMP}.log"

# Run the Python script
python /wynton/protected/group/ibrahim/alex/Dr-LLaVA/experiments/MIMIC-ECG/3_process_ecgs.py > "$LOG_FILE" 2>&1