import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

from radical.asyncflow import LocalExecutionBackend, WorkflowEngine

import parsl
from flowgentic.backend_engines.parsl import ParslEngine
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from flowgentic.backend_engines.radical_asyncflow import AsyncFlowEngine

import multiprocessing


@asynccontextmanager
async def resolve_engine(
	engine_id: str,
	n_of_backend_slots: int,
	observer: Optional[Callable[[Dict[str, Any]], None]] = None,
):
	if engine_id == "asyncflow":
		ctx = multiprocessing.get_context("spawn")

		executor = ProcessPoolExecutor(max_workers=n_of_backend_slots, mp_context=ctx)

		try:
			backend = await LocalExecutionBackend(executor)
			flow = await WorkflowEngine.create(backend)
			yield AsyncFlowEngine(flow, observer=observer)
		finally:
			# 3. Shutdown the flow, then manually shut down the executor
			await flow.shutdown()
			executor.shutdown(wait=True)
	elif engine_id == "parsl":
		parsl_config = Config(
    		executors=[HighThroughputExecutor(max_workers_per_node=n_of_backend_slots, label="local", encrypted=False)]
		)
		try:
			yield ParslEngine(config=parsl_config, observer=observer)
		finally:
			parsl.dfk().cleanup()		
	else:
		raise Exception(f"Didnt match any engine for engine_id: {engine_id}")
