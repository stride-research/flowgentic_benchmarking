import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from data_generation.experiments.base.base_plots import BasePlotter
from data_generation.utils.io_utils import DiscordNotifier

# Silence matplotlib's verbose font manager DEBUG logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _extract_event_durations(events: List[Dict]) -> Dict[str, List[float]]:
	"""
	Match start/end events by ID and compute durations.

	Returns dict with keys:
	- 'task_wrap': list of tool wrapping durations  (tool_wrap_start/end)
	- 'block_wrap': list of block wrapping durations (block_wrap_start/end)
	- 'task_exec': list of tool invocation durations (tool_invoke_start/end)
	"""
	starts = {}
	ends = {}

	for e in events:
		event_type = e["event"]

		if event_type == "tool_wrap_start":
			starts[("task_wrap", e["wrap_id"])] = e["ts"]
		elif event_type == "tool_wrap_end":
			ends[("task_wrap", e["wrap_id"])] = e["ts"]
		elif event_type == "tool_invoke_start":
			starts[("task_exec", e["invocation_id"])] = e["ts"]
		elif event_type == "tool_invoke_end":
			ends[("task_exec", e["invocation_id"])] = e["ts"]
		elif event_type == "block_wrap_start":
			starts[("block_wrap", e["wrap_id"])] = e["ts"]
		elif event_type == "block_wrap_end":
			ends[("block_wrap", e["wrap_id"])] = e["ts"]

	durations = {"task_wrap": [], "block_wrap": [], "task_exec": []}

	for key, start_ts in starts.items():
		if key in ends:
			duration = ends[key] - start_ts
			durations[key[0]].append(duration)

	return durations


def _aggregate_records_by_slots(
	records: List[Dict],
) -> Tuple[List[int], List[float], List[float], List[Dict]]:
	"""
	Group records by n_of_backend_slots and compute mean ± std of total_makespan.

	Returns:
		(backend_slots, mean_makespans, std_makespans, representative_records)
		representative_records: one record per slot count (first of each group).
	"""
	from collections import defaultdict

	groups: Dict[int, List[Dict]] = defaultdict(list)
	for r in records:
		groups[r["n_of_backend_slots"]].append(r)

	backend_slots = sorted(groups.keys())
	mean_makespans = []
	std_makespans = []
	representative_records = []

	for slots in backend_slots:
		group = groups[slots]
		makespans = [r["total_makespan"] for r in group]
		mean_makespans.append(float(np.mean(makespans)))
		std_makespans.append(float(np.std(makespans, ddof=1)) if len(makespans) > 1 else 0.0)
		representative_records.append(group[0])

	return backend_slots, mean_makespans, std_makespans, representative_records


def _compute_overhead_metrics(records: List[Dict]) -> Dict[str, Any]:
	"""
	Compute overhead metrics, aggregated by backend_slots across iterations.

	Returns dict with parallel lists indexed by unique slot count:
	- backend_slots
	- makespans_mean/std, total_compilation_time_mean/std
	- task_wrap_times_mean/std, block_wrap_times_mean/std
	- mean_exec_duration_mean/std
	- exec_durations (pooled list per slot count, for box plots)
	- total_tasks_mean
	"""
	from collections import defaultdict

	per_record: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
	for r in records:
		durations = _extract_event_durations(r["events"])
		task_wrap_total = sum(durations["task_wrap"])
		block_wrap_total = sum(durations["block_wrap"])
		per_record[r["n_of_backend_slots"]].append({
			"makespan": r["total_makespan"],
			"compilation": task_wrap_total + block_wrap_total,
			"task_wrap": task_wrap_total,
			"block_wrap": block_wrap_total,
			"exec_durations": durations["task_exec"],
			"mean_exec": float(np.mean(durations["task_exec"])) if durations["task_exec"] else 0.0,
			"total_tasks": len(durations["task_exec"]),
		})

	def _mean_std(vals):
		m = float(np.mean(vals))
		s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
		return m, s

	metrics: Dict[str, list] = {
		"backend_slots": [],
		"makespans_mean": [], "makespans_std": [],
		"total_compilation_time_mean": [], "total_compilation_time_std": [],
		"task_wrap_times_mean": [], "task_wrap_times_std": [],
		"block_wrap_times_mean": [], "block_wrap_times_std": [],
		"mean_exec_duration_mean": [], "mean_exec_duration_std": [],
		"exec_durations": [],
		"total_tasks_mean": [],
	}

	for slots in sorted(per_record):
		group = per_record[slots]
		metrics["backend_slots"].append(slots)

		for src, dst in [
			("makespan", "makespans"),
			("compilation", "total_compilation_time"),
			("task_wrap", "task_wrap_times"),
			("block_wrap", "block_wrap_times"),
			("mean_exec", "mean_exec_duration"),
		]:
			m, s = _mean_std([g[src] for g in group])
			metrics[f"{dst}_mean"].append(m)
			metrics[f"{dst}_std"].append(s)

		pooled_exec = []
		for g in group:
			pooled_exec.extend(g["exec_durations"])
		metrics["exec_durations"].append(pooled_exec)
		metrics["total_tasks_mean"].append(float(np.mean([g["total_tasks"] for g in group])))

	return metrics


class SyntheticAdaptivePlotter(BasePlotter):
	"""Handles saving benchmark results and generating plots for synthetic adaptive experiments."""

	def __init__(self, plots_dir: Optional[Path] = None) -> None:
		super().__init__()
		self.plots_dir = plots_dir
		self.discord_notifier = DiscordNotifier()

	def set_plots_dir(self, plots_dir: Path) -> None:
		"""Set the plots directory after initialization."""
		self.plots_dir = plots_dir

	def plot_results(self, data: List[Dict[Any, Any]]) -> None:
		"""
		Generate all plots from experiment data (JSONL list of records).

		Records are grouped by their scaling_key field:
		  'strong_scaling-op-work', 'weak_scaling-op-work', etc.
		"""
		from collections import defaultdict
		logger.debug(f"Received {len(data)} records")

		grouped: Dict[str, List] = defaultdict(list)
		for record in data:
			key = record.get("scaling_key", "unknown")
			grouped[key].append(record)

		for experiment_key, records in grouped.items():
			if experiment_key.startswith("strong_scaling"):
				self._plot_strong_scaling(experiment_key, records)
				self._plot_overhead(experiment_key, records, "strong_scaling")
				self._plot_throughput(experiment_key, records, "strong_scaling")
			elif experiment_key.startswith("weak_scaling"):
				self._plot_weak_scaling(experiment_key, records)
				self._plot_overhead(experiment_key, records, "weak_scaling")
				self._plot_throughput(experiment_key, records, "weak_scaling")

	def _plot_strong_scaling(
		self, experiment_name: str, records: List[Dict[Any, Any]]
	) -> None:
		"""
		Generate strong scaling plots: speedup and efficiency.

		Strong scaling: fixed workload, increasing backend slots.
		Uses relative parallelism (p/p_min) so the baseline doesn't need to
		start at p=1. This correctly handles experiments that exclude small
		slot counts.

		- Speedup = T(p_min) / T(p)
		- Relative parallelism factor = p / p_min
		- Efficiency = Speedup / (p / p_min)
		"""
		if not records:
			logger.warning(f"No records for {experiment_name}, skipping plots.")
			return

		# Aggregate by backend slots: handles n_runs > 1 → mean ± std
		backend_slots, makespans, makespan_stds, repr_records = _aggregate_records_by_slots(records)

		# Calculate speedup and efficiency using relative parallelism
		p_min = backend_slots[0]
		t_baseline = makespans[0]
		speedups = [t_baseline / t_p for t_p in makespans]
		relative_p = [p / p_min for p in backend_slots]
		efficiencies = [s / rp for s, rp in zip(speedups, relative_p)]

		# Propagate errors: speedup = t_baseline / t_p
		# sigma_speedup = speedup * sqrt((sigma_base/t_base)^2 + (sigma_p/t_p)^2)
		has_errors = any(s > 0 for s in makespan_stds)
		speedup_errors = makespan_errors = efficiency_errors = None
		if has_errors:
			makespan_errors = makespan_stds
			speedup_errors = [
				sp * np.sqrt((makespan_stds[0] / t_baseline) ** 2 + (s_p / t_p) ** 2)
				for sp, t_p, s_p in zip(speedups, makespans, makespan_stds)
			]
			efficiency_errors = [se / rp for se, rp in zip(speedup_errors, relative_p)]

		# Get metadata for titles
		n_agents = repr_records[0].get("n_of_agents", "?")
		n_tools = repr_records[0].get("n_of_tool_calls_per_agent", "?")
		n_runs = len([r for r in records if r.get("n_of_backend_slots") == backend_slots[0]])
		N_total = (
			n_agents * n_tools
			if isinstance(n_agents, int) and isinstance(n_tools, int)
			else "?"
		)
		n_runs = repr_records[0].get("number_of_runs", 1)

		makespan_subdir = "strong_scaling/makespan"
		baseline_note = f"p₀={p_min}" if p_min > 1 else ""
		subtitle = f"N={N_total}, {n_agents} agents × {n_tools} tools/agent"
		if baseline_note:
			subtitle += f", {baseline_note}"
		if n_runs > 1:
			subtitle += f", {n_runs} runs"

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=speedups,
			title=f"Strong Scaling: Speedup\n({subtitle})",
			xlabel="Number of Backend Slots (p)",
			ylabel=f"Speedup (T(p₀)/T(p))",
			filename="speedup.png",
			subdirectory=makespan_subdir,
			ideal_line=relative_p,
			ideal_label="Ideal (linear)",
			y_errors=speedup_errors,
		)

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=efficiencies,
			title=f"Strong Scaling: Efficiency\n({subtitle})",
			xlabel="Number of Backend Slots (p)",
			ylabel="Efficiency (Speedup / (p/p₀))",
			filename="efficiency.png",
			subdirectory=makespan_subdir,
			ideal_line=[1.0] * len(backend_slots),
			ideal_label="Ideal (100%)",
			y_max=1.1,
			y_errors=efficiency_errors,
		)

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=makespans,
			title=f"Strong Scaling: Makespan\n({subtitle})",
			xlabel="Number of Backend Slots (p)",
			ylabel="Makespan (seconds)",
			filename="makespan.png",
			subdirectory=makespan_subdir,
			y_errors=makespan_errors,
		)

		logger.info(f"Generated strong scaling makespan plots in {makespan_subdir}/")

	def _plot_weak_scaling(
		self, experiment_name: str, records: List[Dict[Any, Any]]
	) -> None:
		"""
		Generate weak scaling plots: speedup and efficiency.

		Weak scaling: workload increases proportionally with backend slots.
		Uses relative parallelism (p/p_min) so the baseline doesn't need to
		start at p=1.

		- Efficiency = T(p_min) / T(p)  (ideally stays at 1)
		- Scaled Speedup = (p/p_min) * T(p_min) / T(p)
		"""
		if not records:
			logger.warning(f"No records for {experiment_name}, skipping plots.")
			return

		# Aggregate by backend slots: handles n_runs > 1 → mean ± std
		backend_slots, makespans, makespan_stds, repr_records = _aggregate_records_by_slots(records)

		p_min = backend_slots[0]
		t_baseline = makespans[0]
		relative_p = [p / p_min for p in backend_slots]
		efficiencies = [t_baseline / t_p for t_p in makespans]
		scaled_speedups = [rp * t_baseline / t_p for rp, t_p in zip(relative_p, makespans)]

		has_errors = any(s > 0 for s in makespan_stds)
		efficiency_errors = speedup_errors = None
		if has_errors:
			efficiency_errors = [
				eff * np.sqrt((makespan_stds[0] / t_baseline) ** 2 + (s_p / t_p) ** 2)
				for eff, t_p, s_p in zip(efficiencies, makespans, makespan_stds)
			]
			speedup_errors = [
				ss * np.sqrt((makespan_stds[0] / t_baseline) ** 2 + (s_p / t_p) ** 2)
				for ss, t_p, s_p in zip(scaled_speedups, makespans, makespan_stds)
			]

		n_agents = repr_records[0].get("n_of_agents", "?")
		n_runs = len([r for r in records if r.get("n_of_backend_slots") == backend_slots[0]])

		makespan_subdir = "weak_scaling/makespan"
		baseline_note = f", p₀={p_min}" if p_min > 1 else ""
		subtitle = f"{n_agents} agents, workload ∝ p{baseline_note}"
		if n_runs > 1:
			subtitle += f", {n_runs} runs"

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=efficiencies,
			title=f"Weak Scaling: Efficiency\n({subtitle})",
			xlabel="Number of Backend Slots (p)",
			ylabel="Efficiency (T(p₀)/T(p))",
			filename="efficiency.png",
			subdirectory=makespan_subdir,
			ideal_line=[1.0] * len(backend_slots),
			ideal_label="Ideal (100%)",
			y_max=1.1,
			y_errors=efficiency_errors,
		)

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=scaled_speedups,
			title=f"Weak Scaling: Scaled Speedup\n({subtitle})",
			xlabel="Number of Backend Slots (p)",
			ylabel="Scaled Speedup ((p/p₀)·T(p₀)/T(p))",
			filename="speedup.png",
			subdirectory=makespan_subdir,
			ideal_line=relative_p,
			ideal_label="Ideal (linear)",
			y_errors=speedup_errors,
		)

		logger.info(f"Generated weak scaling makespan plots in {makespan_subdir}/")

	def _create_scaling_plot(
		self,
		x_values: List[float],
		y_values: List[float],
		title: str,
		xlabel: str,
		ylabel: str,
		filename: str,
		subdirectory: Optional[str] = None,
		ideal_line: Optional[List[float]] = None,
		ideal_label: str = "Ideal",
		y_max: Optional[float] = None,
		y_errors: Optional[List[float]] = None,
	) -> None:
		"""Create a single scaling plot with optional ideal reference line and error bars."""
		fig, ax = plt.subplots(figsize=(8, 6))

		# Set logarithmic x-axis, then pin ticks to exact data points only
		ax.set_xscale("log")
		ax.xaxis.set_major_locator(mticker.FixedLocator(x_values))
		ax.xaxis.set_major_formatter(
			mticker.FixedFormatter([str(int(x)) for x in x_values])
		)
		ax.xaxis.set_minor_locator(mticker.NullLocator())

		# Plot actual values — error bars when multiple runs were recorded
		has_errors = y_errors is not None and any(e > 0 for e in y_errors)
		if has_errors:
			ax.errorbar(
				x_values, y_values, yerr=y_errors,
				fmt="bo-", linewidth=2, markersize=8, capsize=4,
				capthick=1.5, label="Measured (mean ± std)",
			)
		else:
			ax.plot(x_values, y_values, "bo-", linewidth=2, markersize=8, label="Measured")

		# Plot ideal line if provided
		if ideal_line is not None:
			ax.plot(
				x_values, ideal_line, "r--", linewidth=1.5, alpha=0.7, label=ideal_label
			)

		ax.set_xlabel(xlabel, fontsize=12)
		ax.set_ylabel(ylabel, fontsize=12)
		ax.set_title(title, fontsize=14)
		ax.grid(True, alpha=0.3, which="both")
		ax.legend(loc="best")

		if y_max is not None:
			ax.set_ylim(bottom=0, top=y_max)
		else:
			ax.set_ylim(bottom=0)

		plt.tight_layout()

		# Save plot
		if self.plots_dir:
			if subdirectory:
				# Create subdirectory if it doesn't exist
				subdir_path = self.plots_dir / subdirectory
				subdir_path.mkdir(parents=True, exist_ok=True)
				plot_path = subdir_path / filename
			else:
				plot_path = self.plots_dir / filename

			fig.savefig(plot_path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved plot: {plot_path}")

			# Send plot to Discord
			plot_description = (
				f"📊 **{subdirectory}/{filename}**"
				if subdirectory
				else f"📊 **{filename}**"
			)
			try:
				self.discord_notifier.send_discord_notification(
					msg=plot_description, file_path=str(plot_path)
				)
				logger.info(f"Sent plot to Discord: {plot_path}")
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		else:
			logger.warning(f"No plots_dir set, cannot save {filename}")

		plt.close(fig)

	# ==================== OVERHEAD PLOTS ====================

	def _plot_overhead(
		self, experiment_name: str, records: List[Dict[Any, Any]], scaling_type: str
	) -> None:
		"""
		Plot overhead metrics from event data.

		Generates plots in {scaling_type}/overhead/:
		1. compilation_total.png - Total compilation time vs slots
		2. compilation_breakdown.png - Task vs block wrapping time breakdown
		3. exec_overhead_mean.png - Mean per-task execution duration
		4. exec_overhead_distribution.png - Box plot of execution durations
		5. overhead_percentage.png - Framework overhead as % of makespan
		"""
		if not records:
			return

		metrics = _compute_overhead_metrics(records)
		overhead_subdir = f"{scaling_type}/overhead"

		n_agents = records[0].get("n_of_agents", "?")
		n_tools = records[0].get("n_of_tool_calls_per_agent", "?")
		n_runs = len(records) // max(len(metrics["backend_slots"]), 1)
		subtitle = f"({n_agents} agents, {n_tools} tool calls/agent"
		if n_runs > 1:
			subtitle += f", {n_runs} runs"
		subtitle += ")"

		has_errors = any(s > 0 for s in metrics["total_compilation_time_std"])

		self._create_scaling_plot(
			x_values=metrics["backend_slots"],
			y_values=[t * 1000 for t in metrics["total_compilation_time_mean"]],
			title=f"Compilation Overhead\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Total Compilation Time (ms)",
			filename="compilation_total.png",
			subdirectory=overhead_subdir,
			y_errors=[s * 1000 for s in metrics["total_compilation_time_std"]] if has_errors else None,
		)

		self._create_stacked_bar_plot(
			x_values=metrics["backend_slots"],
			y_stacks={
				"Task Wrapping": [t * 1000 for t in metrics["task_wrap_times_mean"]],
				"Block Wrapping": [t * 1000 for t in metrics["block_wrap_times_mean"]],
			},
			title=f"Compilation Overhead Breakdown\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Time (ms)",
			filename="compilation_breakdown.png",
			subdirectory=overhead_subdir,
		)

		self._create_scaling_plot(
			x_values=metrics["backend_slots"],
			y_values=metrics["mean_exec_duration_mean"],
			title=f"Mean Task Execution Duration\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Mean Duration (seconds)",
			filename="exec_overhead_mean.png",
			subdirectory=overhead_subdir,
			y_errors=metrics["mean_exec_duration_std"] if has_errors else None,
		)

		self._create_box_plot(
			data=metrics["exec_durations"],
			labels=[str(s) for s in metrics["backend_slots"]],
			title=f"Task Execution Duration Distribution\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Duration (seconds)",
			filename="exec_overhead_distribution.png",
			subdirectory=overhead_subdir,
		)

		overhead_pct_mean = [
			(comp / ms) * 100 if ms > 0 else 0
			for comp, ms in zip(metrics["total_compilation_time_mean"], metrics["makespans_mean"])
		]
		self._create_scaling_plot(
			x_values=metrics["backend_slots"],
			y_values=overhead_pct_mean,
			title=f"Framework Overhead Percentage\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Overhead (% of makespan)",
			filename="overhead_percentage.png",
			subdirectory=overhead_subdir,
		)

		logger.info(f"Generated overhead plots in {overhead_subdir}/")

	# ==================== THROUGHPUT PLOTS ====================

	def _plot_throughput(
		self, experiment_name: str, records: List[Dict[Any, Any]], scaling_type: str
	) -> None:
		"""
		Plot throughput metrics from event data.

		Generates plots in {scaling_type}/throughput/:
		1. throughput.png - Tasks completed per second vs slots
		2. throughput_per_slot.png - Throughput divided by slot count
		3. throughput_scaling.png - Actual vs ideal throughput scaling
		"""
		if not records:
			return

		metrics = _compute_overhead_metrics(records)
		throughput_subdir = f"{scaling_type}/throughput"

		n_agents = records[0].get("n_of_agents", "?")
		n_tools = records[0].get("n_of_tool_calls_per_agent", "?")
		n_runs = len(records) // max(len(metrics["backend_slots"]), 1)
		subtitle = f"({n_agents} agents, {n_tools} tool calls/agent"
		if n_runs > 1:
			subtitle += f", {n_runs} runs"
		subtitle += ")"

		throughputs = [
			n / ms if ms > 0 else 0
			for n, ms in zip(metrics["total_tasks_mean"], metrics["makespans_mean"])
		]

		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=throughputs,
			title=f"Task Throughput\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Throughput (tasks/second)",
			filename="throughput.png",
			subdirectory=throughput_subdir,
			y_errors=throughput_errors,
		)

		throughput_per_slot = [
			t / s if s > 0 else 0 for t, s in zip(throughputs, backend_slots)
		]
		tps_errors = None
		if throughput_errors is not None:
			tps_errors = [
				te / s if s > 0 else 0 for te, s in zip(throughput_errors, backend_slots)
			]
		self._create_scaling_plot(
			x_values=backend_slots,
			y_values=throughput_per_slot,
			title=f"Throughput per Backend Slot\n{subtitle}",
			xlabel="Number of Backend Slots (p)",
			ylabel="Throughput per Slot (tasks/second/slot)",
			filename="throughput_per_slot.png",
			subdirectory=throughput_subdir,
			y_errors=tps_errors,
		)

		if throughputs[0] > 0:
			baseline_throughput = throughputs[0]
			actual_scaling = [t / baseline_throughput for t in throughputs]
			ideal_scaling = [float(s) for s in backend_slots]

			scaling_errors = None
			if throughput_errors is not None:
				scaling_errors = [te / baseline_throughput for te in throughput_errors]

			self._create_scaling_plot(
				x_values=backend_slots,
				y_values=actual_scaling,
				title=f"Throughput Scaling Factor\n{subtitle}",
				xlabel="Number of Backend Slots (p)",
				ylabel="Scaling Factor (relative to 1 slot)",
				filename="throughput_scaling.png",
				subdirectory=throughput_subdir,
				ideal_line=ideal_scaling,
				ideal_label="Ideal (linear)",
				y_errors=scaling_errors,
			)

		logger.info(f"Generated throughput plots in {throughput_subdir}/")

	# ==================== HELPER PLOT METHODS ====================

	def _create_stacked_bar_plot(
		self,
		x_values: List[float],
		y_stacks: Dict[str, List[float]],
		title: str,
		xlabel: str,
		ylabel: str,
		filename: str,
		subdirectory: Optional[str] = None,
	) -> None:
		"""Create a stacked bar chart."""
		fig, ax = plt.subplots(figsize=(8, 6))

		# Use actual x_values for positioning on log scale
		x_positions = np.array(x_values)
		# Calculate appropriate bar width for log scale (proportional to value)
		if len(x_positions) > 1:
			width = x_positions * 0.3  # Width proportional to x value
		else:
			width = x_positions[0] * 0.3

		bottom = np.zeros(len(x_values))

		colors = plt.cm.Set2.colors
		for i, (label, values) in enumerate(y_stacks.items()):
			ax.bar(
				x_positions,
				values,
				width=width,
				label=label,
				bottom=bottom,
				color=colors[i % len(colors)],
			)
			bottom += np.array(values)

		# Set logarithmic x-axis, then pin ticks to exact data points only
		ax.set_xscale("log")
		ax.xaxis.set_major_locator(mticker.FixedLocator(list(x_positions)))
		ax.xaxis.set_major_formatter(
			mticker.FixedFormatter([str(int(x)) for x in x_values])
		)
		ax.xaxis.set_minor_locator(mticker.NullLocator())
		ax.set_xlabel(xlabel, fontsize=12)
		ax.set_ylabel(ylabel, fontsize=12)
		ax.set_title(title, fontsize=14)
		ax.legend(loc="best")
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		self._save_plot(fig, filename, subdirectory)
		plt.close(fig)

	def _create_box_plot(
		self,
		data: List[List[float]],
		labels: List[str],
		title: str,
		xlabel: str,
		ylabel: str,
		filename: str,
		subdirectory: Optional[str] = None,
	) -> None:
		"""Create a box plot for distribution visualization."""
		fig, ax = plt.subplots(figsize=(8, 6))

		# Filter out empty lists
		filtered_data = []
		filtered_labels = []
		filtered_positions = []
		for d, label in zip(data, labels):
			if d:
				filtered_data.append(d)
				filtered_labels.append(label)
				# Convert label to numeric value for log positioning
				try:
					filtered_positions.append(float(label))
				except ValueError:
					# If label is not numeric, use index
					filtered_positions.append(len(filtered_positions) + 1)

		if not filtered_data:
			plt.close(fig)
			return

		# Use numeric positions for log scale
		bp = ax.boxplot(filtered_data, positions=filtered_positions, patch_artist=True)

		# Style the boxes
		for patch in bp["boxes"]:
			patch.set_facecolor("lightblue")
			patch.set_alpha(0.7)

		# Set logarithmic x-axis, then pin ticks to exact data points only
		ax.set_xscale("log")
		ax.xaxis.set_major_locator(mticker.FixedLocator(filtered_positions))
		ax.xaxis.set_major_formatter(mticker.FixedFormatter(filtered_labels))
		ax.xaxis.set_minor_locator(mticker.NullLocator())
		ax.set_xlabel(xlabel, fontsize=12)
		ax.set_ylabel(ylabel, fontsize=12)
		ax.set_title(title, fontsize=14)
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		self._save_plot(fig, filename, subdirectory)
		plt.close(fig)

	def _save_plot(
		self, fig: plt.Figure, filename: str, subdirectory: Optional[str] = None
	) -> None:
		"""Save a plot to the configured directory and send to Discord."""
		if self.plots_dir:
			if subdirectory:
				subdir_path = self.plots_dir / subdirectory
				subdir_path.mkdir(parents=True, exist_ok=True)
				plot_path = subdir_path / filename
			else:
				plot_path = self.plots_dir / filename

			fig.savefig(plot_path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved plot: {plot_path}")

			# Send plot to Discord
			plot_description = (
				f"📊 **{subdirectory}/{filename}**"
				if subdirectory
				else f"📊 **{filename}**"
			)
			try:
				self.discord_notifier.send_discord_notification(
					msg=plot_description, file_path=str(plot_path)
				)
				logger.info(f"Sent plot to Discord: {plot_path}")
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		else:
			logger.warning(f"No plots_dir set, cannot save {filename}")
