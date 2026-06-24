#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=bg_NylC_paper
#SBATCH --output=/work/%u/slurm/logs/bg_NylC_paper/bg_NylC_paper_%j.out
#SBATCH --error=/work/%u/slurm/logs/bg_NylC_paper/bg_NylC_paper_%j.err
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=arthur.hehner@rwth-aachen.de
#SBATCH -A thes2304
#SBATCH --ntasks=1

set -euo pipefail

echo "===== JOB START ====="
date
hostname
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

mkdir -p /work/$USER/logs
mkdir -p /work/$USER/cache/huggingface
mkdir -p /work/$USER/workdir

module purge

#source ~/miniforge3/bin/activate
#conda activate bg

export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate bg

export NVIDIA_LIB_DIRS="$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -type d -name lib | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

cd ~/ba_nylon/boltzgen

NAME="NylC_paper"
CONFIG="NylC/run/NylC_paper.yaml"
NUM_DESIGNS=2000 
BUDGET=100
OUT="/work/$USER/workdir/nylc_run_${NAME}_${SLURM_JOB_ID}"

echo "===== PYTHON CHECK ====="
which python
python --version
which boltzgen

echo "===== ENV CHECK ====="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "CUDA_ROOT: ${CUDA_ROOT:-unset}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"

echo "===== CUDA CHECK ====="
srun --ntasks=1 --gpus-per-task=1 nvidia-smi

srun --ntasks=1 --gpus-per-task=1 python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available")

print("GPU:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))
PY
echo "===== CUEQUIVARIANCE IMPORT CHECK ====="
python - <<'PY'
try:
    import cuequivariance_ops_torch
    print("cuequivariance_ops_torch import OK")
except Exception:
    print("cuequivariance_ops_torch import FAILED")
    raise
PY

echo "===== CHECKING CONFIG ====="
boltzgen check "$CONFIG"

mkdir -p "$OUT"

echo "===== RUNNING BOLTZGEN ====="

srun --ntasks=1 --gpus-per-task=1 \
  boltzgen run "$CONFIG" \
    --output "$OUT" \
    --protocol protein-redesign \
    --num_designs "$NUM_DESIGNS" \
    --budget "$BUDGET"

echo "===== JOB FINISHED ====="
date