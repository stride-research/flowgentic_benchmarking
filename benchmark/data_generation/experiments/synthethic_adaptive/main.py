from typing import Any, Dict, List, Literal, Optional

import yaml
from pathlib import Path

from data_generation.experiments.base.base_experiment import (
	BaseExperiment,
)
from data_generation.experiments.synthethic_adaptive.utils.plots import (
	SyntheticAdaptivePlotter,
)
from data_generation.utils.schemas import (
	BenchmarkConfig,
	BenchmarkedRecord,
	WorkloadConfig,
	WorkloadResult,
)
from data_generation.workload.langgraph import LangraphWorkload

import logging
import os
import requests

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("config.yml")


logger = logging.getLogger(__name__)


ScalingType = Literal["strong", "weak"]


def send_discord_notifaction(msg: str):
	webhook_url = os.getenv("DISCORD_WEBHOOK")
	data = {"content": msg}
	requests.post(webhook_url, json=data)


class SynthethicAdaptive(BaseExperiment):
	def __init__(
		self, benchmark_config: BenchmarkConfig, data_dir: str, plots_dir: str
	) -> None:
		super().__init__(data_dir, plots_dir)
		self.benchmark_config = benchmark_config
		self.plotter = SyntheticAdaptivePlotter(plots_dir=plots_dir)
		self._load_experiment_config()

	def _load_experiment_config(self):
		"""Read experiment-specific parameters from the synthetic_adaptive section."""
		import math
		with open(CONFIG_PATH) as f:
			raw = yaml.safe_load(f)
		exp_cfg = raw.get("synthetic_adaptive", {})

		self.n_of_agents: int = exp_cfg.get("n_of_agents", 2)
		self.n_of_tool_calls_per_agent: int = exp_cfg.get("n_of_tool_calls_per_agent", 2)
		self.tool_execution_duration_time: int = exp_cfg.get("tool_execution_duration_time", 2)

		max_backend_slots: int = exp_cfg.get("max_backend_slots", 8)
		max_exp = int(math.log2(max_backend_slots))
		start_exp = 4 if max_backend_slots >= 16 else 0
		# e.g. max_backend_slots=8  → [1, 2, 4, 8]
		#      max_backend_slots=64 → [16, 32, 64]
		self.backend_slots_sweep: List[int] = [2**i for i in range(start_exp, max_exp + 1)]

	async def _run_scaling_experiment(
		self,
		scaling_type: ScalingType,
		scaling_key: str,
		index: Optional[int] = None,
	) -> None:
		"""
		Generic scaling experiment runner.

		Args:
			scaling_type: "strong" (fixed workload) or "weak" (workload scales with p)
			scaling_key: Grouping key written into each JSONL record (e.g. "strong_scaling-op-work")
			index: If set, run only backend_slots_sweep[index] (dragon mode).
		"""
		cfg = self.benchmark_config
		logger.info(f"=== {scaling_type.upper()} SCALING: {cfg.run_name} ===")

		sweep = self.backend_slots_sweep
		if scaling_type == "strong":
			sweep = list(reversed(sweep))

		if index is not None:
			sweep = [sweep[index]]

		# Weak scaling: workload scales with p — precompute per-slot ratio
		p_max = max(self.backend_slots_sweep)
		reference_N = self.n_of_agents * self.n_of_tool_calls_per_agent
		workload_per_slot = max(1, reference_N // p_max)

		for backend_slots in sweep:
			logger.info(f"\n--- Testing p={backend_slots} backend slots ---")

			if scaling_type == "strong":
				n_tool_calls = self.n_of_tool_calls_per_agent
				n_agents = self.n_of_agents
			else:
				n_tool_calls = workload_per_slot
				n_agents = backend_slots

			workload_config = WorkloadConfig(
				n_of_agents=n_agents,
				n_of_tool_calls_per_agent=n_tool_calls,
				n_of_backend_slots=backend_slots,
				tool_execution_duration_time=self.tool_execution_duration_time,
				engine_id=cfg.engine_id,
			)

			workload_result: WorkloadResult = await self.run_workload(
				workload_orchestrator=LangraphWorkload,
				workload_config=workload_config,
			)
			logger.debug(f"Workload result is: {workload_result}")

			record = BenchmarkedRecord(
				run_name=cfg.run_name,
				run_description=cfg.run_description,
				workload_id=cfg.workload_id,
				engine_id=cfg.engine_id,
				n_of_agents=n_agents,
				n_of_tool_calls_per_agent=n_tool_calls,
				n_of_backend_slots=backend_slots,
				tool_execution_duration_time=self.tool_execution_duration_time,
				scaling_key=scaling_key,
				total_makespan=workload_result.total_makespan,
				events=workload_result.events,
			).model_dump(mode="json")
			logger.debug(f"Writing to logs: {record}")

			self.store_data_to_disk(record)

			send_discord_notifaction(
				f"🚀 **Iteration Complete: {cfg.run_name}**\n"
				f"**Type:** `{scaling_type.upper()}` | **Slots (p):** `{backend_slots}`\n"
				f"**Agents:** {n_agents} | **Calls/Agent:** {n_tool_calls}\n"
				f"⏱️ **Makespan:** `{workload_result.total_makespan:.2f}s`"
			)

	async def run_strong_scaling(self, index: Optional[int] = None) -> None:
		"""Strong scaling: fixed workload, increasing backend slots."""
		is_noop = self.tool_execution_duration_time == 0
		scaling_key = f"strong_scaling-{'noop' if is_noop else 'op'}-work"
		await self._run_scaling_experiment("strong", scaling_key, index=index)

	async def run_weak_scaling(self, index: Optional[int] = None) -> None:
		"""Weak scaling: workload scales proportionally with backend slots."""
		is_noop = self.tool_execution_duration_time == 0
		scaling_key = f"weak_scaling-{'noop' if is_noop else 'op'}-work"
		await self._run_scaling_experiment("weak", scaling_key, index=index)

	async def run_experiment(self, index: Optional[int] = None) -> None:
		"""Run both strong and weak scaling.

		When index is provided (dragon mode), runs only backend_slots_sweep[index]
		for both scaling types in a single dragon invocation.
		"""
		await self.run_strong_scaling(index=index)
		await self.run_weak_scaling(index=index)

	def generate_plots(self, data: List[Dict[Any, Any]]):
		self.plotter.plot_results(data=data)
