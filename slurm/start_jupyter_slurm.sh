#!/bin/bash
set -euo pipefail

PARTITION="c23ms"
TIME="04:00:00"
CPUS="8"
MEM="32G"
PORT="8888"
PROJECT_DIR="$HOME/ba_nylon"
ENV_PATH="$PROJECT_DIR/envs/ba_nylc"

mkdir -p "$PROJECT_DIR/logs"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=jupyter_lab
#SBATCH --output=$PROJECT_DIR/logs/jupyter_%j.out
#SBATCH --error=$PROJECT_DIR/logs/jupyter_%j.err
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --partition=$PARTITION

cd "$PROJECT_DIR"
source "$ENV_PATH/bin/activate"

jupyter lab --no-browser --ip=127.0.0.1 --port=$PORT --port-retries=0
EOF
)

echo "Submitted Jupyter job: $JOB_ID"
echo
echo "Check status:"
echo "squeue -j $JOB_ID"
echo
echo "Get compute node once job is running:"
echo "NODE=\$(squeue -j $JOB_ID -h -o '%N')"
echo "echo \$NODE"
echo
echo "Check Jupyter token:"
echo "cat $PROJECT_DIR/logs/jupyter_${JOB_ID}.err"
echo
echo "Start tunnel from login node to compute node:"
echo "ssh -4 -N -L 8899:localhost:$PORT <NODE>"
echo
echo "Then use in VS Code:"
echo "http://localhost:8899/?token=<TOKEN>"
echo
echo "Stop job later with:"
echo "scancel $JOB_ID"