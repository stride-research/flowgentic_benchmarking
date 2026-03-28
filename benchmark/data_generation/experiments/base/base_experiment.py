from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional
import logging

from data_generation.utils.schemas import WorkloadConfig, WorkloadResult
from data_generation.workload.base_workload import BaseWorkload
from data_generation.workload.utils.engine import resolve_engine

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
	"""
	Each experiment is responsible of:
		1) Generating experiment data (run_experiment)
		2) Store results that data (store_results)
		3) Generating plot (generate_plots)

	"""

	def __init__(self, data_dir, plots_dir) -> None:
		super().__init__()
		self.data_dir = data_dir
		self.plots_dir = plots_dir

	async def run_workload(
		self, workload_orchestrator: BaseWorkload, workload_config: WorkloadConfig
	) -> WorkloadResult:
		# Event collector for profiling
		events: List[Dict[str, Any]] = []

		# Single workload with shared backend across all agents
		workload: BaseWorkload = workload_orchestrator(workload_config=workload_config)
		async with resolve_engine(
			engine_id=workload_config.engine_id,
			n_of_backend_slots=workload_config.n_of_backend_slots,
			observer=events.append,  # Simple observer: just append to list
		) as engine:
			makespan = await workload.run(engine)

		return WorkloadResult(total_makespan=makespan, events=events)

	@abstractmethod
	async def run_experiment(self, index: Optional[int] = None) -> None:
		"""Run experiment. Pass index to run a single sweep point (dragon mode)."""
		pass

	@abstractmethod
	def generate_plots(self, data: List[Dict[Any, Any]]):
		pass

	def finalize(self):
		"""Read data from disk and generate plots."""
		data = self.load_data_from_disk()
		self.generate_plots(data)

	def store_data_to_disk(self, record: Dict[str, Any]) -> None:
		"""Append a single record to data.jsonl (one JSON object per line)."""
		with open(self.data_dir / "data.jsonl", "a") as f:
			f.write(json.dumps(record) + "\n")
		logger.info(f"✓ Record appended to {self.data_dir / 'data.jsonl'}")

	def load_data_from_disk(self) -> List[Dict[Any, Any]]:
		"""Load all records from data.jsonl."""
		path = self.data_dir / "data.jsonl"
		if not path.exists():
			return []
		with open(path) as f:
			return [json.loads(line) for line in f if line.strip()]
