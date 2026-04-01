#!/bin/bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
CONFIG="scripts/delta/debug/interactive/config.yml"

echo "=== Debug smoke test on $(hostname) at $(date) ==="
echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NNODES=$SLURM_NNODES ==="

module load cray-mpich-abi

source ~/ve/rhapsody-cray/bin/activate
echo "Python: $(python --version)"
echo "Dragon: $(which dragon)"

set -a; source .env; set +a

echo "--- throughput_saturation sweep=0 iter=0 ---"
dragon data_generation/run_experiments.py \
    --config "$CONFIG" \
    --experiment throughput_saturation \
    --sweep-index 0 \
    --iter 0

echo "--- syntethic_adaptive sweep=0 iter=0 ---"
dragon data_generation/run_experiments.py \
    --config "$CONFIG" \
    --experiment syntethic_adaptive \
    --sweep-index 0 \
    --iter 0

echo "=== Generating plots ==="
python data_generation/run_experiments.py --config "$CONFIG" --plots-only

echo "=== Smoke test done at $(date) ==="
