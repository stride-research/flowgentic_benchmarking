from enum import Enum
from typing import Any, Callable, Dict, Literal


from data_generation.experiments.base.base_experiment import (
	BaseExperiment,
)
from data_generation.experiments.base.base_plots import BasePlotter
from data_generation.experiments.backend_comparison.utils.plots import (
    BackendComparisonPlotter,
)
from data_generation.utils.schemas import (
	BenchmarkConfig,
	BenchmarkedRecord,
	EngineIDs,
	WorkloadConfig,
	WorkloadResult,
)
from data_generation.workload.langgraph import LangraphWorkload

import logging
import os
import requests

from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)


ENGINES = [EngineIDs.ASYNCFLOW.value, EngineIDs.PARSL.value]


def send_discord_notifaction(msg: str):
	webhook_url = os.getenv("DISCORD_WEBHOOK")
	data = {"content": msg}
	requests.post(webhook_url, json=data)


class BackendComparison(BaseExperiment):
	def __init__(
		self, benchmark_config: BenchmarkConfig, data_dir: str, plots_dir: str
	) -> None:
		super().__init__(data_dir, plots_dir)
		self.benchmark_config = benchmark_config
		self.plotter = BackendComparisonPlotter(plots_dir=plots_dir)
		self.results: Dict[str, Any] = {}

	async def _run_for_engine(
		self,
		config: BenchmarkConfig,
		engine_id: str,
	) -> None:
		"""
		Run the identical workload on a specific engine (Parsl or AsyncFlow).

		Args:
			config: Benchmark configuration
			engine_id: Identifier of the engine to run on ("asyncflow" or "parsl")
		"""
		logger.info(f"=== BACKEND COMPARISON: {engine_id} ===")
		logger.info(f"Config is: {config.model_dump_json(indent=4)}")

		workload_config = WorkloadConfig(
			n_of_agents=config.n_of_agents,
			n_of_tool_calls_per_agent=config.n_of_tool_calls_per_agent,
			n_of_backend_slots=config.n_of_backend_slots,
			tool_execution_duration_time=config.tool_execution_duration_time,
			engine_id=engine_id,
		)

		workload_result: WorkloadResult = await self.run_workload(
			workload_orchestrator=LangraphWorkload,
			workload_config=workload_config,
		)
		logger.debug(f"Workload result is: {workload_result}")

		benchmark_result = BenchmarkedRecord(
			# Metadata
			run_name=config.run_name,
			run_description=config.run_description,
			workload_id=config.workload_id,
			n_of_agents=config.n_of_agents,
			n_of_tool_calls_per_agent=config.n_of_tool_calls_per_agent,
			n_of_backend_slots=config.n_of_backend_slots,
			workload_type=config.workload_type,
			tool_execution_duration_time=config.tool_execution_duration_time,
			# Results
			total_makespan=workload_result.total_makespan,
			events=workload_result.events,
		).model_dump(mode="json")
		logger.debug(f"Writing to logs: {benchmark_result}")

		msg = (
			f"🚀 **Iteration Complete: {config.run_name}**\n"
			f"**Engine:** `{engine_id}`\n"
			f"**Agents:** {config.n_of_agents} | **Calls/Agent:** {config.n_of_tool_calls_per_agent}\n"
			f"⏱️ **Makespan:** `{workload_result.total_makespan:.2f}s`"
		)
		send_discord_notifaction(msg)

        # Write to disk
		self.results[engine_id] = benchmark_result
		self.store_data_to_disk(self.results)

	async def run_experiment(self) -> None:
		"""Run the same workload on both AsyncFlow and Parsl backends"""
		for engine_id in ENGINES:
			await self._run_for_engine(self.benchmark_config, engine_id)

	def generate_plots(self, data: Dict[Any, Any]):
		self.plotter.plot_results(data=data)
