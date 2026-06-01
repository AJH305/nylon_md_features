#!/bin/bash
#SBATCH --job-name=nylc_boltz_variants
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-35
#SBATCH --output=/work/%u/slurm/logs/lab_data_structure_generation/nylc_boltz_%A_%a.out
#SBATCH --error=/work/%u/slurm/logs/lab_data_structure_generation/nylc_boltz_%A_%a.err
#SBATCH --mail-type=BEGIN,END,ERROR
#SBATCH --mail-user=arthur.hehner@rwth-aachen.de
#SBATCH -A thes2304

set -euo pipefail

mkdir -p /work/$USER/slurm/logs/lab_data_structure_generation
mkdir -p /work/$USER/nylc_variant_structures

source ~/miniforge3/bin/activate
conda activate bg

export BOLTZ_CACHE=/home/dwp46550/.boltz

FASTAS=(/home/dwp46550/ba_nylon/inputs/variant_fastas/*.fasta)

FASTA=${FASTAS[$SLURM_ARRAY_TASK_ID]}
VARIANT=$(basename "$FASTA" .fasta)

OUTDIR=/work/$USER/nylc_variant_structures/$VARIANT
mkdir -p "$OUTDIR"

echo "Running variant: $VARIANT"
echo "Input FASTA: $FASTA"
echo "Output: $OUTDIR"

boltz predict "$FASTA" \
  --out_dir "$OUTDIR" \
  --cache /home/dwp46550/.boltz \
  --accelerator gpu \
  --use_msa_server \
  --override