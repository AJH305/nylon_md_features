#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=nylc_boltzgen
#SBATCH --output=/work/%u/logs/nylc_boltzgen_%j.out
#SBATCH --error=/work/%u/logs/nylc_boltzgen_%j.err
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

echo "===== PYTHON CHECK ====="
which python
python --version
python -c "import ssl; print('OpenSSL:', ssl.OPENSSL_VERSION)"
which boltzgen
boltzgen --help | head -20

echo "===== CUDA CHECK ====="
nvidia-smi

python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. Stopping job.")

print("GPU:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))
PY

run_boltzgen () {
    local name="$1"
    local config="$2"
    local num_designs="$3"
    local budget="$4"

    local out="/work/$USER/workdir/nylc_run_${name}_${SLURM_JOB_ID}"

    echo
    echo "========================================"
    echo "STARTING BOLTZGEN: $name"
    echo "Config: $config"
    echo "Output: $out"
    echo "Num designs: $num_designs"
    echo "Budget: $budget"
    echo "========================================"

    if [[ ! -f "$config" ]]; then
        echo "ERROR: Config not found: $config"
        exit 1
    fi

    echo "===== CHECKING CONFIG: $name ====="
    boltzgen check "$config"

    mkdir -p "$out"

    echo "===== RUNNING DESIGN: $name ====="
    boltzgen run "$config" \
      --output "$out" \
      --protocol protein-anything \
      --num_designs "$num_designs" \
      --budget "$budget"

    echo "===== FINISHED: $name ====="
    date
}

run_boltzgen "HP_pocket"     "NylC/run/HP_pocket.yaml"    30 2
run_boltzgen "NylC_pocket"   "NylC/run/NylC_pocket.yaml"  60 2
run_boltzgen "NylC_double"   "NylC/run/NylC_double.yaml"  90 2
run_boltzgen "NylC_negative" "NylC/run/NylC_negativ.yaml" 30 2
run_boltzgen "NylC_paper"    "NylC/run/NylC_paper.yaml"   90 2

echo
echo "===== JOB FINISHED ====="
date