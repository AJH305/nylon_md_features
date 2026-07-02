#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --job-name=bg_NylC_array
#SBATCH --mail-type=END,ERROR
#SBATCH --mail-user=arthur.hehner@rwth-aachen.de
#SBATCH --ntasks=1

set -euo pipefail
set -x

echo "===== JOB START ====="
date
hostname

echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <design_spec> <outdir> <num_designs_per_job> <conda_environment> [extra boltzgen args...]"
    exit 1
fi

DESIGN_SPEC="$1"
OUTDIR="$2"
NUM_DESIGNS="$3"
CONDA_ENVIRONMENT="$4"
shift 4
EXTRA_ARGS=("$@")

mkdir -p /work/$USER/logs
mkdir -p /work/$USER/cache/huggingface
mkdir -p /work/$USER/workdir

module purge

export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate "$CONDA_ENVIRONMENT"

export NVIDIA_LIB_DIRS="$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -type d -name lib | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

cd ~/ba_nylon/boltzgen

TASK_OUT="${OUTDIR}/task-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TASK_OUT"

echo "===== SETTINGS ====="
echo "DESIGN_SPEC: $DESIGN_SPEC"
echo "OUTDIR: $OUTDIR"
echo "TASK_OUT: $TASK_OUT"
echo "NUM_DESIGNS: $NUM_DESIGNS"
echo "CONDA_ENVIRONMENT: $CONDA_ENVIRONMENT"
echo "EXTRA_ARGS: ${EXTRA_ARGS[*]:-none}"

echo "===== PYTHON CHECK ====="
which python
python --version
which boltzgen

echo "===== ENV CHECK ====="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "CUDA_ROOT: ${CUDA_ROOT:-unset}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

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
boltzgen check "$DESIGN_SPEC"

echo "===== RUNNING BOLTZGEN ====="

srun --ntasks=1 --gpus-per-task=1 \
    boltzgen run "$DESIGN_SPEC" \
    --output "$TASK_OUT" \
    --num_designs "$NUM_DESIGNS" \
    "${EXTRA_ARGS[@]}"

echo "success" > "${TASK_OUT}/_SUCCESS.txt"
date >> "${TASK_OUT}/_SUCCESS.txt"

echo "===== JOB FINISHED ====="
date