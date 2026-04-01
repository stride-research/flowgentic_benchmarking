.PHONY: install install-benchmark install-all \
	smoke-throughput smoke-scaling \
	local-throughput local-scaling local-plots \
	delta-setup delta-debug delta-submit delta-plots \
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

# LOCAL
local-throughput: ## [phase 2] Full local throughput sweep  (ITERS=1)
	$(VENV_ACTIVATE) && python scripts/local/run.py \
		--experiment throughput_saturation --iters $(or $(ITERS),1)

local-scaling: ## [phase 2] Full local scaling sweep  (ITERS=1)
	$(VENV_ACTIVATE) && python scripts/local/run.py \
		--experiment syntethic_adaptive --iters $(or $(ITERS),1)

local-plots: ## [phase 2] Regenerate local plots from existing data
	$(VENV_ACTIVATE) && python scripts/local/run.py --plots-only

# DELTA
delta-setup: ## [phase 3] One-time Dragon/RHAPSODY setup (run once per venv)
	bash scripts/delta/utils/setup.sh
## DEBUG
### INTERACTIVE
delta-debug-interactive-job: ## [phase 3] Submit lightweight smoke test job to Delta
	bash scripts/delta/debug/interactive/debug.sh

delta-debug-interactive-session: ## [phase 3] Start interactive CPU session on Delta for setup/debugging
	salloc --account=bebo-delta-cpu \
		--partition=cpu-interactive \
		--nodes=2 \
		--ntasks-per-node=4 \
		--time=01:00:00
### BATCH
delta-debug-batch-job: ## [phase 3] Submit lightweight smoke test job to Delta
	sbatch scripts/delta/debug/batch/debug.sh

# FULL JOB
delta-submit: ## [phase 4] Submit full benchmark job to Delta
	sbatch scripts/delta/submit.sh

delta-plots: ## [phase 4] Regenerate Delta plots from existing data (no dragon needed)
	$(VENV_ACTIVATE) && python -m data_generation.run_experiments \
		--config scripts/delta/config.yml --plots-only