#!/bin/bash
#SBATCH --job-name=flowgentic-N8192-p2048
#SBATCH --account=bebo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/N8192_p2048_%j.out
#SBATCH --error=logs/N8192_p2048_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
CONFIG="scripts/delta/config.yml"
mkdir -p logs

echo "=== Starting on $(hostname) at $(date) ==="
echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NNODES=$SLURM_NNODES ==="

module load cray-mpich-abi

source ~/ve/rhapsody-cray/bin/activate
echo "Python: $(python --version)"
echo "Dragon: $(which dragon)"

set -a; source .env; set +a
echo "DISCORD_WEBHOOK is ${DISCORD_WEBHOOK:+set}"

N_THROUGHPUT=$(python3 -c "import yaml; print(len(yaml.safe_load(open('$CONFIG'))['throughput_saturation']['n_of_agents_sweep']))")
N_SYNTHETIC=$(python3 -c "import yaml; print(len(yaml.safe_load(open('$CONFIG'))['synthetic_adaptive']['backend_slots_sweep']))")
N_ITERS=1

echo "=== throughput_saturation: $N_THROUGHPUT sweep points ==="
for sweep_index in $(seq 0 $((N_THROUGHPUT - 1))); do
    echo "--- throughput sweep=$sweep_index iter=0 ---"
    dragon data_generation/run_experiments.py \
        --config "$CONFIG" \
        --experiment throughput_saturation \
        --sweep-index "$sweep_index" \
        --iter 0
done

echo "=== syntethic_adaptive: $N_SYNTHETIC sweep points × $N_ITERS iters ==="¶
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
