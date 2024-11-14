#!/bin/bash
#$ -cwd                    # Use the current working directory
#$ -j yes                   # Use the current working directory
#$ -q gpu.q
#$ -pe smp 1               # slots (threads)
#$ -l gpu_mem=75G        # Gigabytes of memory per thread (total 20 * 10G = 200 G)
#$ -R y
#$ -V
#$ -l h_rt=100:50:00        # job time

export PATH=$PATH:~/miniconda3/envs/LLM_env/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/wynton/protected/home/ibrahim/alex_schubert/miniconda3/lib
source /wynton/protected/home/ibrahim/alex_schubert/miniconda3/etc/profile.d/conda.sh
set -e
set -x
module load cuda/11.5

# # Activate the virtual environment
conda activate LLM_env

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
# Note: below 8888 is used to signify the port.
#       However, it may be another number if 8888 is in use.
#       Check jupyter_notebook_%j.err to find the port.

# Command to create SSH tunnel:
ssh -N -f -L 8888:${node}:8888 ${user}@plog1.wynton.ucsf.edu

# Use a browser on your local machine to go to:
http://localhost:8888/
"

jupyter lab --no-browser --ip=${node}
