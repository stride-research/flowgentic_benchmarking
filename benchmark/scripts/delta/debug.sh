#!/bin/bash
# Lightweight smoke test — run interactively before submitting the full job.
# Validates Dragon setup, venv, OFI config, and the full data → plot pipeline
# with minimal resources (1 sweep point, 1 iter, small backend).
#
# Usage (from an interactive allocation or login node):
#   salloc --account=bebo-delta-cpu --partition=cpu-interactive \
#          --nodes=1 --ntasks-per-node=4 --time=00:30:00
#   bash scripts/delta/debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/debug_config.yml"

cd "$REPO_ROOT"
mkdir -p logs

echo "=== Debug smoke test on $(hostname) at $(date) ==="

module load cray-mpich-abi

source ~/ve/rhapsody-cray/bin/activate
echo "Python: $(python --version)"
echo "Dragon: $(which dragon)"

dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

set -a; source .env; set +a

# One sweep point, one iter — just enough to exercise the full pipeline.
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
