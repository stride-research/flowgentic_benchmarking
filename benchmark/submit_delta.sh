#!/bin/bash
# ── Resource allocation ────────────────────────────────────────────────────────
# throughput_saturation: n_of_backend_slots=512 workers → 512/128 = 4 nodes min.
# Using 4 nodes (512 CPUs) exactly matches the slot count.
# Adjust --nodes and n_of_backend_slots together if you change the sweep config.
#SBATCH --job-name=flowgentic-throughput-saturation
#SBATCH --account=bebo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/throughput_%j.out
#SBATCH --error=logs/throughput_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== Starting on $(hostname) at $(date) ==="
echo "=== Working directory: $(pwd) ==="
echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_NNODES=$SLURM_NNODES ==="

# ── Activate venv (already installed on shared fs) ────────────────────────────
source .venv/bin/activate
echo "Python: $(python --version)"
echo "Dragon: $(which dragon)"

# ── Load .env (Discord webhook etc.) ──────────────────────────────────────────
set -a
source .env
set +a
echo "DISCORD_WEBHOOK is ${DISCORD_WEBHOOK:+set}"

# ── Write config ───────────────────────────────────────────────────────────────
cat > config.yml << EOF
run_name: "delta-throughput-$(date +%Y%m%d_%H%M%S)"
run_description: "FlowGentic throughput saturation sweep (noop tools)"
workload_id: langgraph_asyncflow
engine_id: asyncflow_dragon

throughput_saturation:
  tool_execution_duration_time: 0
  n_of_tool_calls_per_agent: 64
  n_of_backend_slots: 512
  agent_sweep: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
EOF

echo "=== Config ==="
cat config.yml

# ── Run each sweep point as a separate dragon invocation ──────────────────────
# Dragon does not release pool memory until its main process exits.
# One dragon call per index ensures full resource cleanup between points.
AGENT_SWEEP=(1 2 4 8 16 32 64 128 256 512)
N_SWEEP=${#AGENT_SWEEP[@]}

for index in $(seq 0 $((N_SWEEP - 1))); do
    echo "=== dragon: throughput_saturation index=$index (n_agents=${AGENT_SWEEP[$index]}) ==="
    dragon data_generation/run_experiments.py \
        --experiment throughput_saturation \
        --index "$index"
done

# ── Generate plots (no dragon needed) ─────────────────────────────────────────
echo "=== Generating plots ==="
python data_generation/run_experiments.py \
    --experiment throughput_saturation \
    --plots-only

echo "=== Done at $(date) ==="
