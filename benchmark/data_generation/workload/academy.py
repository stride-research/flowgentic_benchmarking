import asyncio
import logging
import time
from typing import Callable

from academy.agent import Agent, action
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager

from flowgentic.agent_orchestration_frameworks.academy import AcademyOrchestrator
from flowgentic.backend_engines.base import BaseEngine

from data_generation.utils.schemas import WorkloadConfig
from data_generation.workload.base_workload import BaseWorkload


logger = logging.getLogger(__name__)


class HpcAgent(Agent):
	"""Academy agent that invokes an HPC-backed tool N times."""

	def __init__(self, fetch_fn: Callable, n_calls: int) -> None:
		self._fetch_fn = fetch_fn
		self._n_calls = n_calls

	@action
	async def run_tasks(self) -> None:
		for _ in range(self._n_calls):
			await self._fetch_fn(location="SFO")


class AcademyWorkload(BaseWorkload):
	def __init__(self, workload_config: WorkloadConfig) -> None:
		super().__init__(workload_config=workload_config)

	async def run(self, engine: BaseEngine) -> float:
		t_execution_start = time.perf_counter()

		orchestrator = AcademyOrchestrator(engine)

		@orchestrator.hpc_task
		async def fetch_temperature(location: str = "SFO") -> dict:
			logger.debug("Executing fetch_temperature tool")
			await asyncio.sleep(self.tool_execution_duration_time)
			return {"temperature": 70, "location": location}

		manager = await Manager.from_exchange_factory(
			factory=LocalExchangeFactory(),
			executors=None,
		)

		async with manager:
			handles = await asyncio.gather(
				*[
					manager.launch(
						HpcAgent,
						args=(fetch_temperature, self.n_of_tool_calls_per_agent),
					)
					for _ in range(self.n_of_agents)
				]
			)
			await asyncio.gather(*[handle.run_tasks() for handle in handles])

		t_execution_end = time.perf_counter()

		return t_execution_end - t_execution_start
