#!/bin/bash
set -euo pipefail
set -x

DESIGN_SPEC="$HOME/ba_nylon/boltzgen/NylC/run/NylC_paper.yaml"

RUN_NAME="NylC_paper"
MERGED_OUT="/work/$USER/workdir/nylc_run_${RUN_NAME}"

NUM_TASKS=160
NUM_DESIGNS_PER_TASK=100

CONDA_ENVIRONMENT="bg"
ACCOUNT="thes2304"
TIME="24:00:00"

BUDGET=160
PROTOCOL="protein-redesign"

OUT="${MERGED_OUT}/task-outputs"
LOGS="${MERGED_OUT}/task-logs"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {submit|process}"
    exit 1
fi

MODE="$1"

if [[ "$MODE" == "submit" ]]; then

    mkdir -p "$OUT"
    mkdir -p "$LOGS"
    mkdir -p "/work/$USER/cache/huggingface"
    mkdir -p "/work/$USER/workdir"

    sbatch \
        -A "$ACCOUNT" \
        -t "$TIME" \
        --export=ALL \
        --array=1-"$NUM_TASKS" \
        -o "$LOGS/stdout.%A-%a.log" \
        -e "$LOGS/stderr.%A-%a.log" \
        boltzgen_NylC_paper_array.sh \
        "$DESIGN_SPEC" \
        "$OUT" \
        "$NUM_DESIGNS_PER_TASK" \
        "$CONDA_ENVIRONMENT" \
        --protocol "$PROTOCOL" \
        --budget "$BUDGET"

    squeue --me

elif [[ "$MODE" == "process" ]]; then

    export CONDA_ROOT="$HOME/miniforge3"
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    export PATH="$CONDA_ROOT/bin:$PATH"
    conda activate "$CONDA_ENVIRONMENT"

    export NVIDIA_LIB_DIRS="$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -type d -name lib | tr '\n' ':')"
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    export HF_HOME="/work/$USER/cache/huggingface"
    export TRANSFORMERS_CACHE="/work/$USER/cache/huggingface"

    cd "$HOME/ba_nylon/boltzgen"

    echo "===== CHECK SUCCESSFUL TASKS ====="
    find "$OUT" -name "_SUCCESS.txt" | sort || true

    echo "Number of successful tasks:"
    find "$OUT" -name "_SUCCESS.txt" | wc -l

    echo "Expected tasks:"
    echo "$NUM_TASKS"

    echo "===== MERGING TASK OUTPUTS ====="
    boltzgen merge "$OUT"/task-* --output "$MERGED_OUT"

    echo "===== FILTERING MERGED OUTPUT ====="
    boltzgen run "$DESIGN_SPEC" \
        --steps filtering \
        --protocol "$PROTOCOL" \
        --output "$MERGED_OUT"

    echo "===== PROCESSING FINISHED ====="
    date

else
    echo "Usage: $0 {submit|process}"
    exit 1
fi