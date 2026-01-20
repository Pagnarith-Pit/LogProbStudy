#!/bin/bash
#SBATCH --job-name=indirect_score
#SBATCH --output=indirect_score_%j.out
#SBATCH --error=indirect_score_%j.err

#SBATCH -p gpu-a100-short
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --time=0-00:30:00
# Load required modules
module purge
module load foss/2022a CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1
module load Python/3.10.4
module load mpi4py/3.1.4

# Run your Python code via uv in parallel
echo "Using Qwen Model"
time srun -n 8 uv run python logprob_scoring.py -m Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

# echo "Using Phi4 Model"
# time srun -n 8 uv run python logprob_scoring.py -m microsoft/Phi-4-reasoning-plus

# echo "Using GPT Model"
# time srun -n 8 uv run python logprob_scoring.py -m openai/gpt-oss-20b

# echo "Using Llama Model"
# time srun -n 8 uv run python logprob_scoring.py -m meta-llama/Llama-3.2-3B-Instruct

# echo "Using Mistral Model"
# time srun -n 8 uv run python logprob_scoring.py -m mistralai/Ministral-3-14B-Reasoning-2512