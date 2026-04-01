#!/usr/bin/env python3
"""Local benchmark runner. Run from benchmark/ root: python scripts/local/run.py"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CONFIG = SCRIPT_DIR / "config.yml"


def read_sweep_lengths() -> dict[str, int]:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    return {
        "syntethic_adaptive": len(cfg["synthetic_adaptive"]["backend_slots_sweep"]),
        "throughput_saturation": len(cfg["throughput_saturation"]["n_of_agents_sweep"]),
    }


def run(experiment: str, sweep_index: int, iter_index: int):
    cmd = [
        sys.executable, "-m", "data_generation.run_experiments",
        "--config", str(CONFIG),
        "--experiment", experiment,
        "--sweep-index", str(sweep_index),
        "--iter", str(iter_index),
    ]
    print(f">>> {experiment}  sweep={sweep_index}  iter={iter_index}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_plots(experiment: str | None = None):
    cmd = [
        sys.executable, "-m", "data_generation.run_experiments",
        "--config", str(CONFIG),
        "--plots-only",
    ]
    if experiment:
        cmd += ["--experiment", experiment]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Local benchmark runner")
    parser.add_argument("--experiment", default=None, help="Run only this experiment")
    parser.add_argument("--iters", type=int, default=1, help="Repetitions per sweep point")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate plots")
    args = parser.parse_args()

    if args.plots_only:
        run_plots(args.experiment)
        return

    sweep_lengths = read_sweep_lengths()
    experiments = [args.experiment] if args.experiment else list(sweep_lengths.keys())

    for exp in experiments:
        n_sweep = sweep_lengths[exp]
        for sweep_index in range(n_sweep):
            for iter_index in range(args.iters):
                run(exp, sweep_index, iter_index)

    run_plots(args.experiment)


if __name__ == "__main__":
    main()
