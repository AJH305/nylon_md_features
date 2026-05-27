#!/bin/bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=test_%j.out

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source ~/venvs/bg/bin/activate

nvidia-smi

python -c "
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
"