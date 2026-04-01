from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class TaskState(BaseModel):
	task_id: int


class WorkloadType(str, Enum):
	FIXED_AGENTS_VARY_TOOLS = "fixed_agents_vary_tools"
	FIXED_TOOLS_VARY_AGENTS = "fixed_tools_vary_agents"


class EngineIDs(str, Enum):
	ASYNCFLOW_LOCAL = "asyncflow_local"
	ASYNCFLOW_DRAGON = "asyncflow_dragon"


class BenchmarkConfig(BaseModel):
	"""Global run metadata parsed from the top-level config.yml fields."""

	run_name: str
	run_description: str
	workload_id: str
	engine_id: str = EngineIDs.ASYNCFLOW_LOCAL.value


class WorkloadConfig(BaseModel):
	n_of_agents: int
	n_of_tool_calls_per_agent: int
	n_of_backend_slots: int
	tool_execution_duration_time: int
	engine_id: str


class WorkloadResult(BaseModel):
	"""Raw metrics/result values from a workload run."""

	total_makespan: float
	events: List[Dict[str, Any]]  # Profiling events from the engine


class BenchmarkedRecord(BaseModel):
	"""Full experiment record: global metadata + per-iteration workload config + results."""

	# Global run metadata
	run_name: str
	run_description: str
	workload_id: str
	engine_id: str
	# Per-iteration workload configuration (set by each experiment)
	n_of_agents: int
	n_of_tool_calls_per_agent: int
	n_of_backend_slots: int
	workload_type: WorkloadType = WorkloadType.FIXED_AGENTS_VARY_TOOLS
	tool_execution_duration_time: int
	# JSONL grouping key — used by multi-sub-experiment types (e.g. "strong_scaling-op-work")
	scaling_key: Optional[str] = None
	# Repetition index — 0-based, used when n_runs > 1 to get mean ± std in plots
	run_index: int = 0
	# Results
	total_makespan: float
<<<<<<< HEAD:data_generation/utils/schemas.py
	events: List[Dict[str, Any]]
=======
	events: List[Dict[str, Any]]  # Profiling events from the engine
	run_index: int = 0
>>>>>>> main:benchmark/data_generation/utils/schemas.py
