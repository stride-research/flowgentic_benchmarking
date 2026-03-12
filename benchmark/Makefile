.PHONY: install format lint docs tests examples-sequential-research examples-sequential-financial examples-supervisor examples-supervisor-sales examples-supervisor-product-research examples-basic examples-parallel-minimal examples-parallel-llm-router examples-mcp-sales-analytics

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

# 3) Benchmark
benchmark: ## Run the experiments in the benchmarking
	$(VENV_ACTIVATE) && python3 -m data_generation.run_experiments

benchmark-multiple-workloads: ## Run the full benchmark suite with multiple configs
	$(VENV_ACTIVATE) && python3 -m run_benchmark_suite

sbatch-start: ## Starts batch for testing
	sbatch tests/benchmark/submit_delta.sh

cluster-debug: ## Starts interactive debug session
	salloc --account=bebo-delta-gpu \
		--partition=gpuA100x4-interactive \
		--nodes=2 \
		--ntasks-per-node=4 \
		--gpus-per-node=1 \
		--time=01:00:00