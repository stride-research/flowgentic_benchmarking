from typing import Any, Dict, List, Literal

import yaml
from pathlib import Path

from data_generation.experiments.base.base_experiment import (
	BaseExperiment,
)
from data_generation.experiments.synthethic_adaptive.utils.plots import (
	SyntheticAdaptivePlotter,
)
from data_generation.utils.io_utils import DiscordNotifier
from data_generation.utils.schemas import (
	BenchmarkConfig,
	BenchmarkedRecord,
	WorkloadConfig,
	WorkloadResult,
)
from data_generation.workload.langgraph import LangraphWorkload

import logging

logger = logging.getLogger(__name__)

ScalingType = Literal["strong", "weak"]


class SynthethicAdaptive(BaseExperiment):
	def __init__(
		self,
		benchmark_config: BenchmarkConfig,
		data_dir: str,
		plots_dir: str,
		config_path: Path = Path("config.yml"),
	) -> None:
		super().__init__(data_dir, plots_dir, config_path)
		self.benchmark_config = benchmark_config
		self.plotter = SyntheticAdaptivePlotter(plots_dir=plots_dir)
		self._load_experiment_config()

	def _load_experiment_config(self):
		"""Read experiment-specific parameters from the synthetic_adaptive section."""
		with open(self.config_path) as f:
			raw = yaml.safe_load(f)
		exp_cfg = raw.get("synthetic_adaptive", {})

		self.n_of_agents: int = exp_cfg.get("n_of_agents", 2)
		self.n_of_tool_calls_per_agent: int = exp_cfg.get("n_of_tool_calls_per_agent", 2)
		self.tool_execution_duration_time: int = exp_cfg.get("tool_execution_duration_time", 2)
		self.backend_slots_sweep: List[int] = exp_cfg.get("backend_slots_sweep", [1, 2, 4, 8])

	async def _run_scaling_experiment(
		self,
		scaling_type: ScalingType,
		scaling_key: str,
		sweep_index: int,
		iter_index: int,
	) -> None:
		cfg = self.benchmark_config

		sweep = self.backend_slots_sweep
		if scaling_type == "strong":
			sweep = list(reversed(sweep))
		backend_slots = sweep[sweep_index]

		logger.info(f"=== {scaling_type.upper()} SCALING: {cfg.run_name}  p={backend_slots}  iter={iter_index} ===")

		# Weak scaling: workload scales with p — precompute per-slot ratio
		p_max = max(self.backend_slots_sweep)
		reference_N = self.n_of_agents * self.n_of_tool_calls_per_agent
		workload_per_slot = max(1, reference_N // p_max)

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
			run_index=iter_index,
			total_makespan=workload_result.total_makespan,
			events=workload_result.events,
		).model_dump(mode="json")
		logger.debug(f"Writing to logs: {record}")

		self.store_data_to_disk(record)

		DiscordNotifier().send_discord_notification(
			msg=(
				f"✅ **Done** `synthetic_adaptive` · {scaling_type} scaling\n"
				f"```\n"
				f"slots (p)   : {backend_slots}\n"
				f"agents      : {n_agents}\n"
				f"calls/agent : {n_tool_calls}\n"
				f"duration    : {self.tool_execution_duration_time}s\n"
				f"iter        : {iter_index}\n"
				f"makespan    : {workload_result.total_makespan:.2f}s\n"
				f"```"
			)
		)

	async def run_strong_scaling(self, sweep_index: int, iter_index: int) -> None:
		"""Strong scaling: fixed workload, increasing backend slots."""
		is_noop = self.tool_execution_duration_time == 0
		scaling_key = f"strong_scaling-{'noop' if is_noop else 'op'}-work"
		await self._run_scaling_experiment("strong", scaling_key, sweep_index=sweep_index, iter_index=iter_index)

	async def run_weak_scaling(self, sweep_index: int, iter_index: int) -> None:
		"""Weak scaling: workload scales proportionally with backend slots."""
		is_noop = self.tool_execution_duration_time == 0
		scaling_key = f"weak_scaling-{'noop' if is_noop else 'op'}-work"
		await self._run_scaling_experiment("weak", scaling_key, sweep_index=sweep_index, iter_index=iter_index)

	async def run_experiment(self, sweep_index: int, iter_index: int) -> None:
		"""Run both strong and weak scaling."""
		await self.run_strong_scaling(sweep_index=sweep_index, iter_index=iter_index)
		await self.run_weak_scaling(sweep_index=sweep_index, iter_index=iter_index)

	def generate_plots(self, data: List[Dict[Any, Any]]):
		self.plotter.plot_results(data=data)
