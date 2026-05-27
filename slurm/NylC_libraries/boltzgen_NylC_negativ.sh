#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=bg_NylC_negative
#SBATCH --output=/work/%u/logs/bg_NylC_negative_%j.out
#SBATCH --error=/work/%u/logs/bg_NylC_negative_%j.err
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=arthur.hehner@rwth-aachen.de

set -euo pipefail

echo "===== JOB START ====="
date
hostname
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

mkdir -p /work/$USER/logs
mkdir -p /work/$USER/cache/huggingface
mkdir -p /work/$USER/workdir

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source ~/venvs/bg/bin/activate

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

cd ~/ba_nylon/boltzgen

NAME="NylC_negative"
CONFIG="NylC/run/NylC_negativ.yaml"
NUM_DESIGNS=30
BUDGET=2
OUT="/work/$USER/workdir/nylc_run_${NAME}_${SLURM_JOB_ID}"

echo "===== PYTHON CHECK ====="
which python
python --version

echo "===== CUDA CHECK ====="
nvidia-smi

python - <<'PY'
import torch

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== CHECKING CONFIG ====="
boltzgen check "$CONFIG"

mkdir -p "$OUT"

echo "===== RUNNING BOLTZGEN ====="

boltzgen run "$CONFIG" \
  --output "$OUT" \
  --protocol protein-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET"

echo "===== JOB FINISHED ====="
date