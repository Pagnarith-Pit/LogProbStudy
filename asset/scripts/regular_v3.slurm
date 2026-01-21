#!/bin/bash
#SBATCH --job-name=indirect_score
#SBATCH --output=indirect_score_%A_%a.out
#SBATCH --error=indirect_score_%A_%a.err

#SBATCH -p gpu-l40s
#SBATCH --nodes=1
#SBATCH --mem=512GB
#SBATCH --gres=gpu:1

#SBATCH --array=0-4

#SBATCH --mail-user=ppit@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --time=0-02:00:00

############################
# Environment setup
############################
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1
module load Python/3.10.4
module load SciPy-bundle/2022.05
source ~/venvs/venv-3.10.4/bin/activate

# HuggingFace cache
export HF_HOME=/data/gpfs/projects/punim2402/.hf_cache
############################
# Model selection via array
############################


MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  "meta-llama/Llama-3.2-3B-Instruct"
  "microsoft/Phi-4-reasoning-plus"
  "openai/gpt-oss-20b"
  "google/gemma-3-12b-it"
)



MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "====================================="
echo "SLURM Job ID      : $SLURM_JOB_ID"
echo "Array Task ID     : $SLURM_ARRAY_TASK_ID"
echo "Running Model     : $MODEL"
echo "Node List         : $SLURM_NODELIST"
echo "GPUs allocated    : $SLURM_GPUS"
echo "====================================="

############################
# Run experiment
# DO NOT run with uv. uv run is wrapping python and breaking the MPI runtime linkage
############################
time srun python logprob_scoring.py -m "$MODEL"
