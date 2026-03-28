import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_generation.experiments.base.base_experiment import (
	BaseExperiment,
)
from data_generation.experiments.throughput_saturation.utils.plots import (
	ThroughputSaturationPlotter,
)
from data_generation.utils.schemas import (
	BenchmarkConfig,
	WorkloadConfig,
	WorkloadResult,
)
from data_generation.workload.langgraph import LangraphWorkload

logger = logging.getLogger(__name__)


def extract_invocation_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Extract per-invocation timing metrics from the event model.

	Groups events by invocation_id and computes:
	  D_resolve  = Ts_resolve_end   - Ts_invoke_start   (FlowGentic registry lookup)
	  D_backend  = Ts_collect_start - Ts_resolve_end    (AsyncFlow execution, ≈0 with noop)
	  D_collect  = Ts_collect_end   - Ts_collect_start  (FlowGentic result handling)
	  D_overhead = D_resolve + D_collect                (total FlowGentic overhead)
	  D_total    = Ts_collect_end   - Ts_invoke_start   (end-to-end per invocation)

	  T_run      = max(Ts_collect_end) - min(Ts_invoke_start)
	  throughput = N_completions / T_run
	"""
	# Group timestamps by invocation_id
	by_id: Dict[str, Dict[str, float]] = {}
	for event in events:
		inv_id = event.get("invocation_id")
		if not inv_id:
			continue
		by_id.setdefault(inv_id, {})
		by_id[inv_id][event["event"]] = event["ts"]

	d_resolve_list = []
	d_backend_list = []
	d_collect_list = []
	d_overhead_list = []
	d_total_list = []
	invoke_starts = []
	collect_ends = []
	cache_hits = []

	for inv_id, ts in by_id.items():
		ts_invoke_start = ts.get("tool_invoke_start")
		ts_resolve_end = ts.get("tool_resolve_end")
		ts_collect_start = ts.get("tool_collect_start")
		ts_collect_end = ts.get("tool_invoke_end")  # tool_invoke_end == Ts_collect_end

		# Only include complete invocations
		if not all([ts_invoke_start, ts_resolve_end, ts_collect_start, ts_collect_end]):
			continue

		d_resolve = ts_resolve_end - ts_invoke_start
		d_backend = ts_collect_start - ts_resolve_end
		d_collect = ts_collect_end - ts_collect_start
		d_overhead = d_resolve + d_collect
		d_total = ts_collect_end - ts_invoke_start

		d_resolve_list.append(d_resolve)
		d_backend_list.append(d_backend)
		d_collect_list.append(d_collect)
		d_overhead_list.append(d_overhead)
		d_total_list.append(d_total)
		invoke_starts.append(ts_invoke_start)
		collect_ends.append(ts_collect_end)

	# Cache hit tracking (from tool_resolve_end events)
	cache_hit_events = [e for e in events if e.get("event") == "tool_resolve_end"]
	n_cache_hits = sum(1 for e in cache_hit_events if e.get("cache_hit"))
	n_total_resolve = len(cache_hit_events)

	n_completions = len(d_total_list)
	if n_completions == 0:
		return {"n_completions": 0}

	def _box(arr):
		return {
			"p5":  float(np.percentile(arr, 5)),
			"p25": float(np.percentile(arr, 25)),
			"p50": float(np.percentile(arr, 50)),
			"p75": float(np.percentile(arr, 75)),
			"p95": float(np.percentile(arr, 95)),
		}

	t_run = max(collect_ends) - min(invoke_starts)
	throughput = n_completions / t_run if t_run > 0 else 0.0

	return {
		"n_completions": n_completions,
		"t_run": t_run,
		"throughput": throughput,
		# D_resolve
		"d_resolve_mean": float(np.mean(d_resolve_list)),
		"d_resolve_p95": float(np.percentile(d_resolve_list, 95)),
		# D_backend (≈0 with noop tools)
		"d_backend_mean": float(np.mean(d_backend_list)),
		"d_backend_p95": float(np.percentile(d_backend_list, 95)),
		# D_collect
		"d_collect_mean": float(np.mean(d_collect_list)),
		"d_collect_p95": float(np.percentile(d_collect_list, 95)),
		# D_overhead = D_resolve + D_collect
		"d_overhead_mean": float(np.mean(d_overhead_list)),
		"d_overhead_p95": float(np.percentile(d_overhead_list, 95)),
		# D_total
		"d_total_mean": float(np.mean(d_total_list)),
		"d_total_p95": float(np.percentile(d_total_list, 95)),
		# Overhead fraction (should be ~1.0 with noop tools)
		"overhead_fraction_mean": float(np.mean(
			[oh / tot for oh, tot in zip(d_overhead_list, d_total_list) if tot > 0]
		)),
		# Cache hits
		"n_cache_hits": n_cache_hits,
		"n_total_resolve": n_total_resolve,
		# Box summary stats for boxplots (p5/p25/p50/p75/p95)
		"d_resolve_box":  _box(d_resolve_list),
		"d_collect_box":  _box(d_collect_list),
		"d_overhead_box": _box(d_overhead_list),
		"d_backend_box":  _box(d_backend_list),
	}


class ThroughputSaturation(BaseExperiment):
	"""
	FlowGentic coordination throughput saturation experiment.

	Uses noop tools (tool_execution_duration_time=0) so D_backend ≈ 0.
	Sweeps n_of_agents to increase concurrent load on FlowGentic's async event loop.
	Measures where FlowGentic's coordination throughput saturates.

	Throughput = N_completions / T_run
	         where T_run = max(Ts_collect_end) - min(Ts_invoke_start)
	"""

	def __init__(
		self,
		benchmark_config: BenchmarkConfig,
		data_dir: str,
		plots_dir: str,
		config_path: Path = Path("config.yml"),
	) -> None:
		super().__init__(data_dir, plots_dir, config_path)
		self.benchmark_config = benchmark_config
		self.plotter = ThroughputSaturationPlotter(plots_dir=plots_dir)
		self._load_experiment_config()

	def _load_experiment_config(self):
		"""Read experiment-specific sweep parameters from config.yml."""
		with open(self.config_path) as f:
			raw = yaml.safe_load(f)
		exp_cfg = raw.get("throughput_saturation", {})

		self.n_of_agents_sweep: List[int] = exp_cfg.get("n_of_agents_sweep", [1, 2, 4, 8, 16, 32, 64, 128])
		self.n_of_tool_calls_per_agent: int = exp_cfg.get("n_of_tool_calls_per_agent", 64)
		self.n_of_backend_slots: int = exp_cfg.get("n_of_backend_slots", 512)
		self.tool_execution_duration_time: int = exp_cfg.get("tool_execution_duration_time", 0)

	async def run_experiment(
		self,
		sweep_index: Optional[int] = None,
		iter_index: Optional[int] = None,
	) -> None:
		sweep = [self.n_of_agents_sweep[sweep_index]] if sweep_index is not None else self.n_of_agents_sweep

		logger.info("=== FLOWGENTIC THROUGHPUT SATURATION (noop tools) ===")
		logger.info(
			f"sweep={sweep}  k={self.n_of_tool_calls_per_agent}  "
			f"S={self.n_of_backend_slots}  D={self.tool_execution_duration_time}"
		)

		for n_agents in sweep:
			total_invocations = n_agents * self.n_of_tool_calls_per_agent
			logger.info(f"\n--- n_agents={n_agents}  total_invocations={total_invocations} ---")

			workload_config = WorkloadConfig(
				n_of_agents=n_agents,
				n_of_tool_calls_per_agent=self.n_of_tool_calls_per_agent,
				n_of_backend_slots=self.n_of_backend_slots,
				tool_execution_duration_time=self.tool_execution_duration_time,
				engine_id=self.benchmark_config.engine_id,
			)

			run_index = iter_index if iter_index is not None else 0
			if iter_index is not None:
				logger.info(f"  Run iter={run_index} for n_agents={n_agents}")

			workload_result: WorkloadResult = await self.run_workload(
				workload_orchestrator=LangraphWorkload,
				workload_config=workload_config,
			)

			metrics = extract_invocation_metrics(workload_result.events)

			logger.info(
				f"    throughput={metrics.get('throughput', 0):.2f} inv/s  "
				f"t_run={metrics.get('t_run', 0):.3f}s  "
				f"d_overhead_mean={metrics.get('d_overhead_mean', 0)*1000:.2f}ms  "
				f"d_total_p95={metrics.get('d_total_p95', 0)*1000:.2f}ms"
			)

			record = {
				"n_agents": n_agents,
				"n_of_tool_calls_per_agent": self.n_of_tool_calls_per_agent,
				"n_of_backend_slots": self.n_of_backend_slots,
				"tool_execution_duration_time": self.tool_execution_duration_time,
				"total_invocations": total_invocations,
				"run_index": run_index,
				"total_makespan": workload_result.total_makespan,
				**metrics,
			}
			self.store_data_to_disk(record)

	def generate_plots(self, data: List[Dict[Any, Any]]):
		self.plotter.plot_results(data)
