import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from data_generation.experiments.base.base_plots import BasePlotter
from data_generation.utils.io_utils import DiscordNotifier

# Silence matplotlib's verbose font manager DEBUG logs
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


ENGINE_COLORS = {
    "asyncflow": "#2196F3",
    "parsl": "#FF9800",
}

ENGINE_LABELS = {
    "asyncflow": "AsyncFlow",
    "parsl": "Parsl",
}

_SET2 = plt.cm.Set2.colors
COMP_COLORS = {
    "D_wrap": _SET2[0],
    "D_overhead": _SET2[1],
    "D_backend": _SET2[2],
}
COMP_LABELS = {
    "D_wrap": "D_wrap (amortized)",
    "D_overhead": "D_overhead (D_resolve + D_collect)",
    "D_backend": "D_backend",
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

	Each tool invocation goes through 4 stages:
	tool_invoke_start -> tool_resolve_end -> tool_collect_start -> tool_invoke_end
	
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

	Returns one entry per engine_id with everything the plots need:
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
		self._plot_invocation_breakdown(metrics, engines, "overhead")
		self._plot_invocation_proportional(metrics, engines, "overhead")
		self._plot_results_table(metrics, engines)
		self._plot_overhead_vs_backend(metrics, engines, "overhead")
		self._plot_overhead_vs_backend_proportional(metrics, engines, "overhead")


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

	def _plot_invocation_breakdown(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Stacked bar: D_wrap (amortized) + D_overhead + D_backend per engine. Absolute mean time per invocation (ms)"""
		fig, ax = plt.subplots(figsize=(8, 6))

		labels = [ENGINE_LABELS.get(e, e) for e in engines]
		x = np.arange(len(engines))
		width = 0.5
		components = ["D_wrap", "D_overhead", "D_backend"]
		values: Dict[str, List[float]] = {c: [] for c in components}

		for e in engines:
			m = metrics[e]
			values["D_wrap"].append(m["d_wrap_amortized_ms"])
			values["D_overhead"].append(np.mean(m["d_overhead"]) * 1000 if m["d_overhead"] else 0.0)
			values["D_backend"].append(np.mean(m["d_backend"]) * 1000 if m["d_backend"] else 0.0)

		bottom = np.zeros(len(engines))
		for comp in components:
			vals = np.array(values[comp])
			ax.bar(x, vals, width, label=COMP_LABELS[comp], bottom=bottom, color=COMP_COLORS[comp])
			bottom += vals

		for i, total in enumerate(bottom):
			ax.text(x[i], total, f"{total:.3f}ms", ha="center", va="bottom", fontsize=10)

		ax.set_xlabel("Backend Engine", fontsize=12)
		ax.set_ylabel("Mean Time per Invocation (ms)", fontsize=12)
		ax.set_title(f"Invocation Breakdown (incl. D_wrap amortized)\n{self._subtitle}", fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend(loc="best")
		ax.grid(True, alpha=0.3, axis="y")
		ax.set_ylim(bottom=0)

		plt.tight_layout()
		self._save_plot(fig, "invocation_breakdown.png", subdir)
		plt.close(fig)

	def _plot_invocation_proportional(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Stacked bar: D_wrap (amortized) + D_overhead + D_backend normalized to 100%"""
		fig, ax = plt.subplots(figsize=(8, 6))

		labels = [ENGINE_LABELS.get(e, e) for e in engines]
		x = np.arange(len(engines))
		width = 0.5
		components = ["D_wrap", "D_overhead", "D_backend"]
		raw: Dict[str, List[float]] = {c: [] for c in components}

		for e in engines:
			m = metrics[e]
			raw["D_wrap"].append(m["d_wrap_amortized_ms"])
			raw["D_overhead"].append(np.mean(m["d_overhead"]) * 1000 if m["d_overhead"] else 0.0)
			raw["D_backend"].append(np.mean(m["d_backend"]) * 1000 if m["d_backend"] else 0.0)

		totals = [sum(raw[c][i] for c in components) for i in range(len(engines))]
		values = {
			c: [raw[c][i] / totals[i] * 100 if totals[i] > 0 else 0 for i in range(len(engines))]
			for c in components
		}

		bottom = np.zeros(len(engines))
		for comp in components:
			vals = np.array(values[comp])
			ax.bar(x, vals, width, label=COMP_LABELS[comp], bottom=bottom, color=COMP_COLORS[comp])
			for i, (v, b) in enumerate(zip(vals, bottom)):
				if v > 5:
					ax.text(x[i], b + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=9)
			bottom += vals

		ax.set_xlabel("Backend Engine", fontsize=12)
		ax.set_ylabel("% of Effective Invocation Cost", fontsize=12)
		ax.set_title(f"Invocation Breakdown Proportional\n{self._subtitle}", fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend(loc="best")
		ax.grid(True, alpha=0.3, axis="y")
		ax.set_ylim(0, 100)

		plt.tight_layout()
		self._save_plot(fig, "invocation_proportional.png", subdir)
		plt.close(fig)
	
	
	def _plot_overhead_vs_backend(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Stacked bar: D_overhead + D_backend per engine. No D_wrap. Total time (ms)"""
		fig, ax = plt.subplots(figsize=(8, 6))

		labels = [ENGINE_LABELS.get(e, e) for e in engines]
		x = np.arange(len(engines))
		width = 0.5
		components = ["D_overhead", "D_backend"]
		values: Dict[str, List[float]] = {c: [] for c in components}

		for e in engines:
			m = metrics[e]
			values["D_overhead"].append(sum(m["d_overhead"]) * 1000 if m["d_overhead"] else 0.0)
			values["D_backend"].append(sum(m["d_backend"]) * 1000 if m["d_backend"] else 0.0)

		bottom = np.zeros(len(engines))
		for comp in components:
			vals = np.array(values[comp])
			ax.bar(x, vals, width, label=COMP_LABELS[comp], bottom=bottom, color=COMP_COLORS[comp])
			bottom += vals

		for i, total in enumerate(bottom):
			ax.text(x[i], total, f"{total:.3f}ms", ha="center", va="bottom", fontsize=10)

		ax.set_xlabel("Backend Engine", fontsize=12)
		ax.set_ylabel("Total Time (ms)", fontsize=12)
		ax.set_title(f"Overhead vs Backend Total (= D_total)\n{self._subtitle}", fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend(loc="best")
		ax.grid(True, alpha=0.3, axis="y")
		ax.set_ylim(bottom=0)

		plt.tight_layout()
		self._save_plot(fig, "overhead_vs_backend.png", subdir)
		plt.close(fig)

	def _plot_overhead_vs_backend_proportional(self, metrics: Dict, engines: List[str], subdir: str) -> None:
		"""Stacked bar: D_overhead + D_backend normalized to 100%. No D_wrap."""
		fig, ax = plt.subplots(figsize=(8, 6))

		labels = [ENGINE_LABELS.get(e, e) for e in engines]
		x = np.arange(len(engines))
		width = 0.5
		components = ["D_overhead", "D_backend"]
		raw: Dict[str, List[float]] = {c: [] for c in components}

		for e in engines:
			m = metrics[e]
			raw["D_overhead"].append(sum(m["d_overhead"]) * 1000 if m["d_overhead"] else 0.0)
			raw["D_backend"].append(sum(m["d_backend"]) * 1000 if m["d_backend"] else 0.0)

		totals = [sum(raw[c][i] for c in components) for i in range(len(engines))]
		values = {
			c: [raw[c][i] / totals[i] * 100 if totals[i] > 0 else 0 for i in range(len(engines))]
			for c in components
		}

		bottom = np.zeros(len(engines))
		for comp in components:
			vals = np.array(values[comp])
			ax.bar(x, vals, width, label=COMP_LABELS[comp], bottom=bottom, color=COMP_COLORS[comp])
			for i, (v, b) in enumerate(zip(vals, bottom)):
				if v > 5:
					ax.text(x[i], b + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=9)
			bottom += vals

		ax.set_xlabel("Backend Engine", fontsize=12)
		ax.set_ylabel("% of Total D_total", fontsize=12)
		ax.set_title(f"Overhead vs Backend Proportional\n{self._subtitle}", fontsize=14)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend(loc="best")
		ax.grid(True, alpha=0.3, axis="y")
		ax.set_ylim(0, 100)

		plt.tight_layout()
		self._save_plot(fig, "overhead_vs_backend_proportional.png", subdir)
		plt.close(fig)

	def _plot_results_table(self, metrics: Dict, engines: List[str]) -> None:
		"""Summary table with totals and per-invocation means, grouped by section."""
		fig, ax = plt.subplots(figsize=(12, 7))
		ax.axis("off")

		stats: Dict[str, Dict] = {}
		for e in engines:
			m = metrics[e]
			overhead_fractions = [
				(r + c) / t
				for r, c, t in zip(m["d_resolve"], m["d_collect"], m["d_total"])
				if t > 0
			]
			stats[e] = {
				"makespan": m["makespan"],
				"n_inv": m["n_inv"],
				# totals
				"d_total_total": sum(m["d_total"]) * 1000,
				"d_backend_total": sum(m["d_backend"]) * 1000,
				"d_overhead_total": sum(m["d_overhead"]) * 1000,
				"d_wrap_total": sum(m["d_wrap"]) * 1000,
				# per-invocation means
				"d_total_mean": np.mean(m["d_total"]) * 1000 if m["d_total"] else 0.0,
				"d_backend_mean": np.mean(m["d_backend"]) * 1000 if m["d_backend"] else 0.0,
				"d_overhead_mean": np.mean(m["d_overhead"]) * 1000 if m["d_overhead"] else 0.0,
				"d_wrap_amortized": m["d_wrap_amortized_ms"],
				"overhead_fraction": np.mean(overhead_fractions) if overhead_fractions else 0.0,
			}

		def _fmt(v: float) -> str:
			return f"{v:.3f}"

		SECTION = True
		DATA = False
		row_defs = [
			("Overview", None, SECTION),
			("Makespan (s)", lambda e: f"{stats[e]['makespan']:.3f}", DATA),
			("N invocations", lambda e: str(stats[e]["n_inv"]), DATA),
			("Totals (sum across all invocations)", None, SECTION),
			("D_total total (ms)", lambda e: _fmt(stats[e]["d_total_total"]), DATA),
			("D_backend total (ms)", lambda e: _fmt(stats[e]["d_backend_total"]), DATA),
			("D_overhead total (ms)", lambda e: _fmt(stats[e]["d_overhead_total"]), DATA),
			("D_wrap total (ms)", lambda e: _fmt(stats[e]["d_wrap_total"]), DATA),
			("Per Invocation (mean)", None, SECTION),
			("D_total mean (ms)", lambda e: _fmt(stats[e]["d_total_mean"]), DATA),
			("D_backend mean (ms)", lambda e: _fmt(stats[e]["d_backend_mean"]), DATA),
			("D_overhead mean (ms)", lambda e: _fmt(stats[e]["d_overhead_mean"]), DATA),
			("D_wrap amortized (ms)", lambda e: _fmt(stats[e]["d_wrap_amortized"]), DATA),
			("Overhead fraction", lambda e: f"{stats[e]['overhead_fraction']:.4f}", DATA),
		]

		col_labels = [ENGINE_LABELS.get(e, e) for e in engines]
		rows = [
			[label] + ([""] * len(engines) if fn is None else [fn(e) for e in engines])
			for label, fn, _ in row_defs
		]

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
		for i in range(len(row_defs)):
			is_section = row_defs[i][2]
			row_i = i + 1  
			if is_section:
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

		ax.set_title(f"Results Summary\n{self._subtitle}", fontsize=14, pad=20)
		plt.tight_layout()
		self._save_plot(fig, "results_table.png", None)
		plt.close(fig)

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
