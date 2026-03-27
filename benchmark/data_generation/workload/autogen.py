import asyncio
import logging
import time

import autogen
from autogen import UserProxyAgent

from flowgentic.agent_orchestration_frameworks.autogen import AutoGenOrchestrator
from flowgentic.backend_engines.base import BaseEngine
from flowgentic.core.models.implementations.dummy.autogen import (
	DummyAutoGenClient,
	create_assistant_with_dummy_model,
)

from data_generation.utils.schemas import WorkloadConfig
from data_generation.workload.base_workload import BaseWorkload

logger = logging.getLogger(__name__)


class AutogenWorkload(BaseWorkload):
	"""
	AutoGen workload with configurable number of agents and tool calls per agent.
	Each agent is an independent AssistantAgent + UserProxyAgent conversation,
	all sharing the same HPC-wrapped tools and running concurrently.
	"""

	def __init__(self, workload_config: WorkloadConfig) -> None:
		super().__init__(workload_config=workload_config)

	async def run(self, engine: BaseEngine) -> float:
		t_execution_start = time.perf_counter()

		orchestrator = AutoGenOrchestrator(engine)

		@orchestrator.hpc_task
		async def fetch_temperature(location: str = "SFO") -> dict:
			"""Fetches temperature of a given city."""
			logger.debug("Executing temperature tool")
			await asyncio.sleep(self.tool_execution_duration_time)
			return {"temperature": 70, "location": location}

		tools = [fetch_temperature]

		async def run_agent():
			assistant = create_assistant_with_dummy_model(
				name="hpc_assistant",
				system_message="You are a helpful assistant. Use available tools to fetch weather data.",
				calls_per_tool=self.n_of_tool_calls_per_agent,
			)
			user_proxy = UserProxyAgent(
				name="hpc_executor",
				human_input_mode="NEVER",
				max_consecutive_auto_reply=10,
				is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
				code_execution_config=False,
			)
			for tool in tools:
				autogen.register_function(
					tool,
					caller=assistant,
					executor=user_proxy,
					description=tool.__doc__ or tool.__name__,
				)
			if hasattr(assistant, "client") and assistant.client:
				assistant.client.register_model_client(model_client_cls=DummyAutoGenClient)

			await user_proxy.a_initiate_chat(
				assistant,
				message="Fetch temperature in SFO. When done, reply TERMINATE.",
			)

		await asyncio.gather(*[run_agent() for _ in range(self.n_of_agents)])

		t_execution_end = time.perf_counter()

		return t_execution_end - t_execution_start
