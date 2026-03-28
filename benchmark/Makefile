.PHONY: install install-benchmark install-all \
	smoke-throughput smoke-scaling \
	local-throughput local-scaling local-plots \
	delta-debug delta-submit delta-plots \
	cluster-session

.DEFAULT_GOAL:= help
SHELL := /bin/bash


# VARIABLES FOR PRETTY PRINT
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
BLUE = \033[34m
RESET = \033[0m

# VARIABLES, GENERAL
VENV_PATH =  ./.venv
VENV_ACTIVATE = source $(VENV_PATH)/bin/activate


# 1) Set-up
help: ## Show this help message
	@printf "$(BLUE) Available commands: $(RESET)\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
install: ## Install core dependencies
	uv venv $(VENV_PATH) --python 3.10
	uv pip install -e "."
install-benchmark: ## Install benchmark + dev dependencies (flowgentic + langgraph + asyncflow)
	uv venv $(VENV_PATH) --python 3.10
	uv pip install -e ".[dev]"
install-all: ## Install all dependencies (frameworks + runtimes + dev)
	uv venv $(VENV_PATH) --python 3.10
	uv pip install -e ".[dev]"

# ── Phase 1: local smoke test ─────────────────────────────────────────────────
# One sweep point, one iter. Validates config → workload → JSONL → plots end-to-end.
smoke-throughput: ## [phase 1] Single-point smoke test for throughput_saturation
	$(VENV_ACTIVATE) && python -m data_generation.run_experiments \
		--config scripts/local/config.yml \
		--experiment throughput_saturation \
		--sweep-index 0 --iter 0

smoke-scaling: ## [phase 1] Single-point smoke test for syntethic_adaptive
	$(VENV_ACTIVATE) && python -m data_generation.run_experiments \
		--config scripts/local/config.yml \
		--experiment syntethic_adaptive \
		--sweep-index 0 --iter 0

# ── Phase 2: full local run ───────────────────────────────────────────────────
# All sweep points, configurable iters (default 1). Use ITERS=3 for error bars.
local-throughput: ## [phase 2] Full local throughput sweep  (ITERS=1)
	$(VENV_ACTIVATE) && python scripts/local/run.py \
		--experiment throughput_saturation --iters $(or $(ITERS),1)

local-scaling: ## [phase 2] Full local scaling sweep  (ITERS=1)
	$(VENV_ACTIVATE) && python scripts/local/run.py \
		--experiment syntethic_adaptive --iters $(or $(ITERS),1)

local-plots: ## [phase 2] Regenerate local plots from existing data
	$(VENV_ACTIVATE) && python scripts/local/run.py --plots-only

# ── Phase 3: Delta smoke test (run on cluster, not locally) ──────────────────
delta-debug: ## [phase 3] Submit lightweight smoke test job to Delta
	sbatch scripts/delta/debug.sh

cluster-session: ## [phase 3] Start interactive CPU session on Delta for setup/debugging
	salloc --account=bebo-delta-cpu \
		--partition=cpu-interactive \
		--nodes=1 --ntasks-per-node=4 \
		--time=01:00:00

# ── Phase 4: full Delta job ───────────────────────────────────────────────────
delta-submit: ## [phase 4] Submit full benchmark job to Delta
	sbatch scripts/delta/submit.sh

delta-plots: ## [phase 4] Regenerate Delta plots from existing data (no dragon needed)
	$(VENV_ACTIVATE) && python -m data_generation.run_experiments \
		--config scripts/delta/config.yml --plots-only