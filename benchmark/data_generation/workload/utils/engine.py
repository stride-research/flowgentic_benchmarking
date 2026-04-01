from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

from radical.asyncflow import ConcurrentExecutionBackend, WorkflowEngine

from flowgentic.backend_engines.radical_asyncflow import AsyncFlowEngine

import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def resolve_engine(
	engine_id: str,
	n_of_backend_slots: int,
	observer: Optional[Callable[[Dict[str, Any]], None]] = None,
):
	logger.debug(f"Using engine: {engine_id}")
	if engine_id == "asyncflow_local":
		ctx = mp.get_context("spawn")
		executor = ProcessPoolExecutor(max_workers=n_of_backend_slots, mp_context=ctx)
		flow = None
		try:
			backend = await ConcurrentExecutionBackend(executor)
			flow = await WorkflowEngine.create(backend)
			yield AsyncFlowEngine(flow, observer=observer)
		finally:
			if flow is not None:
				await flow.shutdown()
			executor.shutdown(wait=True)

	elif engine_id == "asyncflow_dragon":
		from radical.asyncflow import DragonExecutionBackendV3
		flow = None
		try:
			backend = await DragonExecutionBackendV3(num_workers=n_of_backend_slots)
			flow = await WorkflowEngine.create(backend)
			yield AsyncFlowEngine(flow, observer=observer)
		finally:
			if flow is not None:
				await flow.shutdown()

	else:
		raise Exception(f"Didnt match any engine for engine_id: {engine_id}")
