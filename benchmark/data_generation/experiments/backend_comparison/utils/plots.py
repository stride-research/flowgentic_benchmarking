import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from data_generation.experiments.base.base_plots import BasePlotter
from data_generation.utils.io_utils import DiscordNotifier

# Silence matplotlib's verbose font manager DEBUG logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


ENGINE_COLORS = {
    "asyncflow": "#2196F3",
    "parsl":     "#FF9800",
}

ENGINE_LABELS = {
    "asyncflow": "AsyncFlow",
    "parsl":     "Parsl",
}

METRIC_COLORS = {
    "d_resolve": ENGINE_COLORS["asyncflow"],
    "d_collect": ENGINE_COLORS["parsl"],
}

METRIC_LABELS = {
    "d_resolve": "D_resolve",
    "d_collect": "D_collect",
}


def _parse_events(events: List[Dict]) -> tuple:
	"""
	Raw events into wrap timestamps and invocation timestamps.

	Returns:
	- wrap_starts: {(tool_name, wrap_id): ts}
	- wrap_ends: {(tool_name, wrap_id): ts}
	- invocations: {invocation_id: {event_type: ts, ...}}
	"""
	wrap_starts: Dict = {}
	wrap_ends: Dict = {}
	invocations: Dict = {}

	for event in events:
		event_type = event["event"]
		if event_type == "tool_wrap_start":
			wrap_starts[(event["tool_name"], event["wrap_id"])] = event["ts"]
		elif event_type == "tool_wrap_end":
			wrap_ends[(event["tool_name"], event["wrap_id"])] = event["ts"]
		elif event_type in ("tool_invoke_start", "tool_resolve_end",
					        "tool_collect_start", "tool_invoke_end"):
			invocation_id = event["invocation_id"]
			if invocation_id not in invocations:
				invocations[invocation_id] = {}
			invocations[invocation_id][event_type] = event["ts"]
			if "tool_name" in event:
				invocations[invocation_id]["tool_name"] = event["tool_name"]

	return wrap_starts, wrap_ends, invocations


def _extract_event_durations(events: List[Dict]) -> Dict:
	"""
	Compute per-invocation durations from raw events.
	
	Returns:
	- d_wrap_by_tool: {tool_name: [D_wrap durations]}
	- d_wrap: all D_wrap durations combined
    - d_resolve: list of D_resolve durations (tool_resolve_end - tool_invoke_start)
    - d_backend: list of D_backend durations (tool_collect_start - tool_resolve_end)
    - d_collect: list of D_collect durations (tool_invoke_end - tool_collect_start)
    - d_total: list of D_total durations (tool_invoke_end - tool_invoke_start)
    - inv_by_tool: {tool_name: count of invocations}
	"""
	wrap_starts, wrap_ends, invocations = _parse_events(events)

	# D_wrap: duration to wrap/register each tool, grouped by tool name
	d_wrap_by_tool: Dict[str, List[float]] = {}
	for (tool_name, wrap_id), ts_start in wrap_starts.items():
		if (tool_name, wrap_id) in wrap_ends:
			duration = wrap_ends[(tool_name, wrap_id)] - ts_start
			if tool_name not in d_wrap_by_tool:
				d_wrap_by_tool[tool_name] = []
			d_wrap_by_tool[tool_name].append(duration)
			
	# Per-invocation stage durations
	d_resolve, d_backend, d_collect, d_total = [], [], [], []
	inv_by_tool: Dict[str, int] = {}
	required_keys = ("tool_invoke_start", "tool_resolve_end",
					  "tool_collect_start", "tool_invoke_end")

	for invocation in invocations.values():
		if not all(k in invocation for k in required_keys):
			continue

		t_invoke_start = invocation["tool_invoke_start"]
		t_resolve_end = invocation["tool_resolve_end"]
		t_collect_start = invocation["tool_collect_start"]
		t_invoke_end = invocation["tool_invoke_end"]

		d_resolve.append(t_resolve_end - t_invoke_start)
		d_backend.append(t_collect_start - t_resolve_end)
		d_collect.append(t_invoke_end - t_collect_start)
		d_total.append(t_invoke_end - t_invoke_start)

		if "tool_name" in invocation:
			tool_name = invocation["tool_name"]
			inv_by_tool[tool_name] = inv_by_tool.get(tool_name, 0) + 1

	return {
		"d_wrap_by_tool": d_wrap_by_tool,
		"d_wrap": [v for vals in d_wrap_by_tool.values() for v in vals],
		"d_resolve": d_resolve,
		"d_backend": d_backend,
		"d_collect": d_collect,
		"d_total": d_total,
		"inv_by_tool": inv_by_tool,
	}


def _compute_invocation_metrics(data: Dict[str, Dict]) -> Dict[str, Dict]:
	"""
	Run _extract_event_durations for each engine and compute derived metrics.

	Returns one entry per engine_id:
	- All raw durations from _extract_event_durations
	- makespan: total experiment wall time
	- n_inv: number of complete invocations
	- d_overhead: D_resolve + D_collect per invocation (flowgentic overhead)
	- d_wrap_amortized_ms: total D_wrap divided by n_inv, in ms
	"""
	metrics = {}
	for engine_id, record in data.items():
		d = _extract_event_durations(record["events"])

		n_inv = len(d["d_resolve"]) or 1
		d_overhead = [r + c for r, c in zip(d["d_resolve"], d["d_collect"])]
		d_wrap_amortized_ms = 0.0
		for tool, inv_count in d["inv_by_tool"].items():
			if inv_count == 0:
				continue
			tool_wrap_time = sum(d["d_wrap_by_tool"].get(tool, []))
			d_wrap_amortized_ms += tool_wrap_time / inv_count * 1000

		metrics[engine_id] = {
			**d,
			"makespan": record["total_makespan"],
			"n_inv": n_inv,
			"d_overhead": d_overhead,
			"d_wrap_amortized_ms": d_wrap_amortized_ms,
		}
	return metrics


class BackendComparisonPlotter(BasePlotter):
	"""Generates comparison plots for backend-adaptive execution experiment."""

	def __init__(self, plots_dir: Optional[Path] = None) -> None:
		super().__init__()
		self.plots_dir = plots_dir
		self.discord_notifier = DiscordNotifier()
		
	def set_plots_dir(self, plots_dir: Path) -> None:
		"""Set the plots directory after initialization."""
		self.plots_dir = plots_dir    

	def _plot_resolve_collect_distributions(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""4 box plots: D_resolve and D_collect per engine side by side."""
		fig, ax = plt.subplots(figsize=(10, 6))

		box_data = []
		tick_labels = []
		colors = []
		positions = []
		pos = 1
		group_centers = []

		for e in engines:
			group_start = pos
			for key in ("d_resolve", "d_collect"):
				box_data.append([v * 1000 for v in metrics[e][key]])
				tick_labels.append(METRIC_LABELS[key])
				colors.append(METRIC_COLORS[key])
				positions.append(pos)
				pos += 1
			group_centers.append((group_start + pos - 1) / 2.0)
			pos += 1  

		bp = ax.boxplot(
			box_data,
			positions=positions,
			patch_artist=True,
			medianprops={"color": "black", "linewidth": 2},
			widths=0.6,
		)
		for patch, color in zip(bp["boxes"], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.7)

		ax.set_xticks(group_centers)
		ax.set_xticklabels([ENGINE_LABELS.get(e, e) for e in engines], fontsize=12, fontweight="bold")

		legend_handles = [Patch(facecolor=METRIC_COLORS[k], alpha=0.7, label=METRIC_LABELS[k])
						  for k in ("d_resolve", "d_collect")]
		ax.legend(handles=legend_handles, loc="upper right")

		ax.set_ylabel("Duration (ms)", fontsize=11)
		ax.set_title(f"D_resolve & D_collect Distributions per Engine\n{self._subtitle}", fontsize=13)
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		self._save_plot(fig, "resolve_collect_distributions.png", subdir)
		plt.close(fig)

	def _plot_overhead_distribution(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Box plot: D_overhead per engine (2 boxes)."""
		fig, ax = plt.subplots(figsize=(7, 6))

		box_data = [[v * 1000 for v in metrics[e]["d_overhead"]] for e in engines]
		labels = [ENGINE_LABELS.get(e, e) for e in engines]
		colors = [ENGINE_COLORS.get(e, "#999") for e in engines]

		bp = ax.boxplot(
			box_data,
			labels=labels,
			patch_artist=True,
			medianprops={"color": "black", "linewidth": 2},
			widths=0.5,
		)
		for patch, color in zip(bp["boxes"], colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.7)

		ax.set_ylabel("Duration (ms)", fontsize=11)
		ax.set_title(f"D_overhead Distribution per Engine\n(D_resolve + D_collect)\n{self._subtitle}", fontsize=13)
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		self._save_plot(fig, "overhead_distribution.png", subdir)
		plt.close(fig)


	def _plot_makespan(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Total execution time (makespan) per engine."""
		fig, ax = plt.subplots(figsize=(8, 6))

		makespans = [metrics[e]["makespan"] for e in engines]
		colors = [ENGINE_COLORS.get(e, "#999") for e in engines]
		labels = [ENGINE_LABELS.get(e, e) for e in engines]

		bars = ax.bar(labels, makespans, color=colors, width=0.5)
		for bar in bars:
			ax.text(
				bar.get_x() + bar.get_width() / 2, bar.get_height(),
				f"{bar.get_height():.3f}s",
				ha="center", va="bottom", fontsize=10,
			)

		ax.set_xlabel("Backend Engine", fontsize=12)
		ax.set_ylabel("Makespan (seconds)", fontsize=12)
		ax.set_title(f"Makespan Comparison\n{self._subtitle}", fontsize=14)
		ax.grid(True, alpha=0.3, axis="y")
		ax.set_ylim(bottom=0)

		plt.tight_layout()
		self._save_plot(fig, "makespan.png", subdir)
		plt.close(fig)

	def _plot_results_table(self, metrics: Dict, engines: List[str]) -> None:
		"""Summary table with totals and per-invocation means, grouped by section."""
		fig, ax = plt.subplots(figsize=(12, 9))
		ax.axis("off")

		def _mean_ms(values: list) -> float:
			return float(np.mean(values)) * 1000 if values else 0.0

		def _overhead_fraction(m: Dict) -> float:
			d_total = np.asarray(m["d_total"], dtype=float)
			mask = d_total > 0
			if not mask.any():
				return 0.0
			overhead = np.asarray(m["d_resolve"])[mask] + np.asarray(m["d_collect"])[mask]
			return float(np.mean(overhead / d_total[mask]))

		row_defs = [
			("Overview", None),
			("Makespan (s)", [f"{metrics[e]['makespan']:.3f}" for e in engines]),
			("N invocations", [str(metrics[e]["n_inv"]) for e in engines]),
			("Totals (sum across all invocations)", None),
			("D_total total (ms)", [f"{sum(metrics[e]['d_total']) * 1000:.3f}" for e in engines]),
			("D_backend total (ms)", [f"{sum(metrics[e]['d_backend']) * 1000:.3f}" for e in engines]),
			("D_overhead total (ms)", [f"{sum(metrics[e]['d_overhead']) * 1000:.3f}" for e in engines]),
			("D_resolve total (ms)", [f"{sum(metrics[e]['d_resolve']) * 1000:.3f}" for e in engines]),
			("D_collect total (ms)", [f"{sum(metrics[e]['d_collect']) * 1000:.3f}" for e in engines]),
			("D_wrap total (ms)", [f"{sum(metrics[e]['d_wrap']) * 1000:.3f}" for e in engines]),
			("Per Invocation (mean)", None),
			("D_total mean (ms)", [f"{_mean_ms(metrics[e]['d_total']):.3f}" for e in engines]),
			("D_backend mean (ms)", [f"{_mean_ms(metrics[e]['d_backend']):.3f}" for e in engines]),
			("D_overhead mean (ms)", [f"{_mean_ms(metrics[e]['d_overhead']):.3f}" for e in engines]),
			("D_resolve mean (ms)", [f"{_mean_ms(metrics[e]['d_resolve']):.3f}" for e in engines]),
			("D_collect mean (ms)", [f"{_mean_ms(metrics[e]['d_collect']):.3f}" for e in engines]),
			("D_wrap amortized (ms)", [f"{metrics[e]['d_wrap_amortized_ms']:.3f}" for e in engines]),
			("Overhead fraction", [f"{_overhead_fraction(metrics[e]):.4f}" for e in engines]),
		]

		col_labels = [ENGINE_LABELS.get(e, e) for e in engines]
		rows = []
		for label, values in row_defs:
			row = [label] + (values if values is not None else [""] * len(engines))
			rows.append(row)

		table = ax.table(
			cellText=rows,
			colLabels=["Metric"] + col_labels,
			loc="center",
			cellLoc="center",
		)
		table.auto_set_font_size(False)
		table.set_fontsize(10)
		table.scale(1.5, 1.9)

		# Header row
		for j in range(len(engines) + 1):
			cell = table[0, j]
			cell.set_facecolor("#2C3E50")
			cell.set_text_props(color="white", fontweight="bold")

		# Data rows
		data_row_idx = 0
		for i, (_, values) in enumerate(row_defs):
			row_i = i + 1
			if values is None:
				for j in range(len(engines) + 1):
					cell = table[row_i, j]
					cell.set_facecolor("#4A6FA5")
					cell.set_text_props(color="white", fontweight="bold")
					cell.set_edgecolor("#2C3E50")
			else:
				color = "#F2F2F2" if data_row_idx % 2 == 0 else "#FFFFFF"
				for j in range(len(engines) + 1):
					table[row_i, j].set_facecolor(color)
					table[row_i, j].set_edgecolor("#CCCCCC")
				data_row_idx += 1

		ax.set_title(f"Results Summary\n{self._subtitle}", fontsize=14, pad=40)
		plt.tight_layout()
		self._save_plot(fig, "results_table.png", None)
		plt.close(fig)


	def plot_results(self, data: Dict[Any, Any]) -> None:
		"""
		Generate all plots from experiment data.

		Data structure expected:
		{
			'asyncflow': BenchmarkedRecord dict,  # AsyncFlow engine results
			'parsl': BenchmarkedRecord dict,  # Parsl engine results
		}
		"""
		engines = list(data.keys())
		if not engines:
			logger.warning("No data to plot.")
			return

		sample = data[engines[0]]
		self._subtitle = (
			f"({sample['n_of_agents']} agents, "
			f"{sample['n_of_tool_calls_per_agent']} tool calls/agent, "
			f"{sample['n_of_backend_slots']} slots)"
		)

		metrics = _compute_invocation_metrics(data)

		self._plot_makespan(metrics, engines, "makespan")
		self._plot_resolve_collect_distributions(metrics, engines, "overhead")
		self._plot_overhead_distribution(metrics, engines, "overhead")
		self._plot_results_table(metrics, engines)


	def _save_plot(self, fig: plt.Figure, filename: str, subdirectory: Optional[str] = None) -> None:
		"""Save a plot to the configured directory."""
		if self.plots_dir:
			if subdirectory:
				subdir_path = self.plots_dir / subdirectory
				subdir_path.mkdir(parents=True, exist_ok=True)
				plot_path = subdir_path / filename
			else:
				plot_path = self.plots_dir / filename
				plot_path.parent.mkdir(parents=True, exist_ok=True)

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
					msg=plot_description, image_path=str(plot_path)
				)
				logger.info(f"Sent plot to Discord: {plot_path}")
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		else:
			logger.warning(f"No plots_dir set, cannot save {filename}")
