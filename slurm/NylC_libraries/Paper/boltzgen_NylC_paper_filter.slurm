#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=bg_NylC_filter
#SBATCH --output=/work/%u/slurm/logs/bg_NylC_filter_%j.out
#SBATCH --error=/work/%u/slurm/logs/bg_NylC_filter_%j.err
#SBATCH --mail-type=END,ERROR
#SBATCH --mail-user=arthur.hehner@rwth-aachen.de
#SBATCH -A thes2304
#SBATCH --ntasks=1

set -euo pipefail
set -x

DESIGN_SPEC="/home/dwp46550/ba_nylon/boltzgen/NylC/run/NylC_paper.yaml"
OUT="/work/dwp46550/workdir/nylc_run_NylC_paper"
BUDGET=160
PROTOCOL="protein-redesign"

mkdir -p /work/$USER/slurm/logs
mkdir -p /work/$USER/cache/huggingface

module purge

export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate bg

export NVIDIA_LIB_DIRS="$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -type d -name lib | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

cd /home/dwp46550/ba_nylon/boltzgen

echo "===== GPU CHECK ====="
srun --ntasks=1 --gpus-per-task=1 nvidia-smi

srun --ntasks=1 --gpus-per-task=1 python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo "===== FILTERING MERGED DESIGNS ====="
srun --ntasks=1 --gpus-per-task=1 \
  boltzgen run "$DESIGN_SPEC" \
    --steps filtering \
    --protocol "$PROTOCOL" \
    --output "$OUT" \
    --budget "$BUDGET"

echo "===== FILTERING FINISHED ====="
date
