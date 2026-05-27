#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=nylc_boltzgen
#SBATCH --output=/work/%u/logs/nylc_boltzgen_%j.out
#SBATCH --error=/work/%u/logs/nylc_boltzgen_%j.err

set -euo pipefail

echo "===== JOB START ====="
date
hostname

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source ~/venvs/bg/bin/activate

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

mkdir -p /work/$USER/logs
mkdir -p /work/$USER/workdir/nylc_test_run

cd ~/ba_nylon/boltzgen

echo "===== CUDA CHECK ====="
nvidia-smi

python -c "
import torch
print('Torch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
"

echo "===== STARTING BOLTZGEN ====="

boltzgen run NylC/test/NylC_test.yaml \
  --output /work/$USER/workdir/nylc_test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2

echo "===== JOB FINISHED ====="
date