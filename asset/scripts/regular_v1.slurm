#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-00:30:00
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8

# Load required modules
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1
module load Python/3.10.4
module load mpi4py/3.1.4

# Run your Python code via uv in parallel
echo "Using GPT Model"
time srun -n 8 uv run python logprob_scoring.py -m

echo "Using Phi4 Model"
time srun -n 8 uv run python logprob_scoring.py -m 

echo "Using Qwen Model"
time srun -n 8 uv run python logprob_scoring.py -m 

echo "Using Llama Model"
time srun -n 8 uv run python logprob_scoring.py -m 

echo "Using Mistral Model"
time srun -n 8 uv run python logprob_scoring.py -m 

