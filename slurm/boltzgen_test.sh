#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=boltzgen_test
#SBATCH --output=/work/%u/logs/boltzgen_%j.out
#SBATCH --error=/work/%u/logs/boltzgen_%j.err

set -euo pipefail

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source ~/venvs/bg/bin/activate

export HF_HOME=/work/$USER/cache/huggingface
export TRANSFORMERS_CACHE=/work/$USER/cache/huggingface

mkdir -p /work/$USER/logs
mkdir -p /work/$USER/workdir/test_run

cd ~/ba_nylon/boltzgen

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output /work/$USER/workdir/test_run \
  --protocol protein-anything \
  --num_designs 2 \
  --budget 1