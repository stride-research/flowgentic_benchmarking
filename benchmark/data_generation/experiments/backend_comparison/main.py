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
from data_generation.workload.autogen import AutogenWorkload
from data_generation.workload.academy import AcademyWorkload

import logging
import os
import requests
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


WORKLOADS = {
    "langraph": (LangraphWorkload, EngineIDs.ASYNCFLOW.value),
    "parsl":    (LangraphWorkload, EngineIDs.PARSL.value),
    "autogen":  (AutogenWorkload,  EngineIDs.ASYNCFLOW.value),
    "academy":  (AcademyWorkload,  EngineIDs.ASYNCFLOW.value),
}


def send_discord_notifaction(msg: str):
	webhook_url = os.getenv("DISCORD_WEBHOOK")
	data = {"content": msg}
	requests.post(webhook_url, json=data)


class BackendComparison(BaseExperiment):
	def __init__(
		self, benchmark_config: BenchmarkConfig, data_dir: str, plots_dir: str
	) -> None:
		super().__init__(data_dir, Path(plots_dir))
		self.benchmark_config = benchmark_config
		self.plotter = BackendComparisonPlotter(plots_dir=Path(plots_dir))
		self.results: Dict[str, Any] = {}

	async def _run_for_engine(
		self,
		config: BenchmarkConfig,
		engine_id: str,
	) -> Dict:
		workload_cls, hpc_backend_id = WORKLOADS[engine_id]

		logger.info(f"=== BACKEND COMPARISON: {engine_id} ===")
		logger.info(f"Config is: {config.model_dump_json(indent=4)}")

		workload_config = WorkloadConfig(
			n_of_agents=config.n_of_agents,
			n_of_tool_calls_per_agent=config.n_of_tool_calls_per_agent,
			n_of_backend_slots=config.n_of_backend_slots,
			tool_execution_duration_time=config.tool_execution_duration_time,
			engine_id=hpc_backend_id,
		)

		workload_result: WorkloadResult = await self.run_workload(
			workload_orchestrator=workload_cls,
			workload_config=workload_config,
		)
		logger.debug(f"Workload result is: {workload_result}")

		record = BenchmarkedRecord(
			run_name=config.run_name,
			run_description=config.run_description,
			workload_id=config.workload_id,
			n_of_agents=config.n_of_agents,
			n_of_tool_calls_per_agent=config.n_of_tool_calls_per_agent,
			n_of_backend_slots=config.n_of_backend_slots,
			workload_type=config.workload_type,
			tool_execution_duration_time=config.tool_execution_duration_time,
			total_makespan=workload_result.total_makespan,
			events=workload_result.events,
		).model_dump(mode="json")

		msg = (
			f"🚀 **Iteration Complete: {config.run_name}**\n"
			f"**Engine:** `{engine_id}`\n"
			f"**Agents:** {config.n_of_agents} | **Calls/Agent:** {config.n_of_tool_calls_per_agent}\n"
			f"⏱️ **Makespan:** `{workload_result.total_makespan:.2f}s`"
		)
		send_discord_notifaction(msg)

		self.results[engine_id] = record
		self.store_data_to_disk(self.results)
		return record

	async def run_experiment(self) -> None:
		"""Run the workload for each engine sequentially."""
		for engine_id in WORKLOADS:
			self.results[engine_id] = await self._run_for_engine(
				self.benchmark_config, engine_id
			)

	def generate_plots(self, data: Dict[Any, Any]):
		plots_dir = self.plotter.plots_dir

		# Backend comparison: LangGraph on AsyncFlow vs Parsl
		self.plotter.set_plots_dir(plots_dir / "backend_comparison")
		self.plotter.plot_results(
			data={k: data[k] for k in ("langraph", "parsl") if k in data},
			engine_labels={"langraph": "AsyncFlow", "parsl": "Parsl"},
		)

		# Orchestrator comparison: LangGraph vs AutoGen vs Academy, all on AsyncFlow
		self.plotter.set_plots_dir(plots_dir / "orchestrator_comparison")
		self.plotter.plot_results(
			data={k: data[k] for k in ("langraph", "autogen", "academy") if k in data},
			engine_labels={"langraph": "LangGraph", "autogen": "AutoGen", "academy": "Academy"},
		)
