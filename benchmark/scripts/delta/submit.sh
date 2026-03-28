#!/bin/bash
#SBATCH --job-name=flowgentic-benchmark
#SBATCH --account=bebo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yml"

cd "$REPO_ROOT"
mkdir -p logs

echo "=== Starting on $(hostname) at $(date) ==="
echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NNODES=$SLURM_NNODES ==="

# cray-mpich-abi must be loaded before activating the venv so Dragon can find
# the Cray OFI/CXI transport library at runtime.
module load cray-mpich-abi

source ~/ve/rhapsody-cray/bin/activate
echo "Python: $(python --version)"
echo "Dragon: $(which dragon)"

dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

set -a; source .env; set +a
echo "DISCORD_WEBHOOK is ${DISCORD_WEBHOOK:+set}"

# Read sweep lengths from config so the loops stay in sync with it.
N_THROUGHPUT=$(python3 -c "import yaml; print(len(yaml.safe_load(open('$CONFIG'))['throughput_saturation']['n_of_agents_sweep']))")
N_SYNTHETIC=$(python3 -c "import yaml; print(len(yaml.safe_load(open('$CONFIG'))['synthetic_adaptive']['backend_slots_sweep']))")
N_ITERS=3

echo "=== throughput_saturation: $N_THROUGHPUT sweep points × $N_ITERS iters ==="
for sweep_index in $(seq 0 $((N_THROUGHPUT - 1))); do
    for iter in $(seq 0 $((N_ITERS - 1))); do
        echo "--- throughput sweep=$sweep_index iter=$iter ---"
        dragon data_generation/run_experiments.py \
            --config "$CONFIG" \
            --experiment throughput_saturation \
            --sweep-index "$sweep_index" \
            --iter "$iter"
    done
done

echo "=== syntethic_adaptive: $N_SYNTHETIC sweep points × $N_ITERS iters ==="
for sweep_index in $(seq 0 $((N_SYNTHETIC - 1))); do
    for iter in $(seq 0 $((N_ITERS - 1))); do
        echo "--- syntethic_adaptive sweep=$sweep_index iter=$iter ---"
        dragon data_generation/run_experiments.py \
            --config "$CONFIG" \
            --experiment syntethic_adaptive \
            --sweep-index "$sweep_index" \
            --iter "$iter"
    done
done

echo "=== Generating plots ==="
python data_generation/run_experiments.py --config "$CONFIG" --plots-only

echo "=== Done at $(date) ==="
