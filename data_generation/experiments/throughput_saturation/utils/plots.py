import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from data_generation.experiments.base.base_plots import BasePlotter
from data_generation.utils.io_utils import DiscordNotifier

logger = logging.getLogger(__name__)


def _bxp_stats(box: Dict[str, float]) -> Dict[str, Any]:
	"""Convert stored box summary into a matplotlib bxp stats dict."""
	return {
		"med":    box["p50"],
		"q1":     box["p25"],
		"q3":     box["p75"],
		"whislo": box["p5"],
		"whishi": box["p95"],
		"fliers": [],
	}


def _aggregate_records_by_agents(
	records: List[Dict[str, Any]],
) -> tuple:
	"""
	Group records by n_agents and compute mean ± std of throughput.

	Returns:
		(n_agents_vals, mean_throughputs, std_throughputs, representative_records)
	"""
	from collections import defaultdict

	groups: Dict[int, List] = defaultdict(list)
	for r in records:
		groups[r["n_agents"]].append(r)

	agents = sorted(groups.keys())
	means = [float(np.mean([r.get("throughput", 0) for r in groups[n]])) for n in agents]
	stds = [
		float(np.std([r.get("throughput", 0) for r in groups[n]], ddof=1))
		if len(groups[n]) > 1 else 0.0
		for n in agents
	]
	reprs = [groups[n][0] for n in agents]
	return agents, means, stds, reprs


class ThroughputSaturationPlotter(BasePlotter):
	"""
	Plots for the FlowGentic coordination throughput saturation experiment.

	x-axis: n_of_agents (concurrent load on FlowGentic's event loop)

	Plots:
	  1. throughput_vs_agents.png    — line: throughput vs n_agents
	  2. overhead_breakdown.png      — boxplot: D_resolve / D_collect per n_agents
	  3. d_overhead_boxplot.png      — boxplot: D_overhead per n_agents
	  4. overhead_fraction.png       — line: D_overhead / D_total per n_agents
	"""

	def __init__(self, plots_dir: Optional[Path] = None) -> None:
		super().__init__()
		self.plots_dir = plots_dir
		self.discord_notifier = DiscordNotifier()

	def plot_results(self, data: List[Dict[str, Any]]) -> None:
		records: List[Dict[str, Any]] = data
		if not records:
			logger.warning("No throughput_saturation data to plot.")
			return

		records = sorted(records, key=lambda r: r["n_agents"])

		self._plot_throughput(records)
		self._plot_overhead_breakdown(records)
		self._plot_overhead_boxplot(records)
		self._plot_overhead_fraction(records)

	def _plot_throughput(self, records: List[Dict[str, Any]]) -> None:
		"""Throughput (inv/s) vs n_agents — aggregates mean ± std across runs."""
		fig, ax = plt.subplots(figsize=(10, 6))

		n_agents, throughputs, throughput_stds, _ = _aggregate_records_by_agents(records)
		has_errors = any(s > 0 for s in throughput_stds)

		if has_errors:
			ax.errorbar(
				n_agents, throughputs, yerr=throughput_stds,
				fmt="o-", linewidth=2, markersize=8, color="steelblue",
				capsize=4, capthick=1.5, label="Mean ± std",
			)
		else:
			ax.plot(n_agents, throughputs, marker="o", linewidth=2, markersize=8, color="steelblue")

		ax.set_xscale("log", base=2)
		ax.set_xlabel("Number of Agents (concurrent load)", fontsize=13)
		ax.set_ylabel("Throughput (invocations/s)", fontsize=13)
		ax.set_title(
			"FlowGentic Coordination Throughput vs Concurrent Agents\n"
			"(noop tools — D_backend ≈ 0 for FlowGentic isolation)",
			fontsize=13,
		)
		ax.grid(True, alpha=0.3)
		ax.set_ylim(bottom=0)

		plt.tight_layout()
		if self.plots_dir:
			path = self.plots_dir / "throughput_vs_agents.png"
			fig.savefig(path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved: {path}")
			try:
				self.discord_notifier.send_discord_notification(
					msg="📊 **throughput_vs_agents.png**", file_path=str(path)
				)
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		plt.close(fig)

	def _plot_overhead_breakdown(self, records: List[Dict[str, Any]]) -> None:
		"""Boxplot of D_resolve and D_collect side-by-side per n_agents."""
		has_box = all(r.get("d_resolve_box") for r in records)
		if not has_box:
			logger.warning("No box stats found — re-run experiment to generate boxplots.")
			return

		fig, ax = plt.subplots(figsize=(12, 6))

		n = len(records)
		positions_resolve = [i * 3 + 1 for i in range(n)]
		positions_collect  = [i * 3 + 2 for i in range(n)]

		bxp_resolve = [_bxp_stats({k: v * 1000 for k, v in r["d_resolve_box"].items()}) for r in records]
		bxp_collect = [_bxp_stats({k: v * 1000 for k, v in r["d_collect_box"].items()}) for r in records]

		bp1 = ax.bxp(bxp_resolve, positions=positions_resolve, widths=0.8,
		             patch_artist=True, showfliers=False)
		bp2 = ax.bxp(bxp_collect, positions=positions_collect, widths=0.8,
		             patch_artist=True, showfliers=False)

		for patch in bp1["boxes"]:
			patch.set_facecolor("steelblue")
			patch.set_alpha(0.7)
		for patch in bp2["boxes"]:
			patch.set_facecolor("coral")
			patch.set_alpha(0.7)

		tick_pos = [i * 3 + 1.5 for i in range(n)]
		ax.set_xticks(tick_pos)
		ax.set_xticklabels([str(r["n_agents"]) for r in records])
		ax.set_xlabel("Number of Agents", fontsize=13)
		ax.set_ylabel("Duration (ms)", fontsize=13)
		ax.set_title("FlowGentic Overhead Breakdown per Invocation\n(D_overhead = D_resolve + D_collect)", fontsize=13)

		from matplotlib.patches import Patch
		ax.legend(handles=[
			Patch(facecolor="steelblue", alpha=0.7, label="D_resolve (registry lookup)"),
			Patch(facecolor="coral",     alpha=0.7, label="D_collect (result handling)"),
		], fontsize=11)
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		if self.plots_dir:
			path = self.plots_dir / "overhead_breakdown.png"
			fig.savefig(path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved: {path}")
			try:
				self.discord_notifier.send_discord_notification(
					msg="📊 **overhead_breakdown.png**", file_path=str(path)
				)
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		plt.close(fig)

	def _plot_overhead_boxplot(self, records: List[Dict[str, Any]]) -> None:
		"""Boxplot of D_overhead per n_agents — primary FlowGentic metric."""
		has_box = all(r.get("d_overhead_box") for r in records)
		if not has_box:
			logger.warning("No box stats found — re-run experiment to generate boxplots.")
			return

		fig, ax = plt.subplots(figsize=(10, 6))

		positions = list(range(1, len(records) + 1))
		bxp_data = [_bxp_stats({k: v * 1000 for k, v in r["d_overhead_box"].items()}) for r in records]

		bp = ax.bxp(bxp_data, positions=positions, widths=0.6,
		            patch_artist=True, showfliers=False)
		for patch in bp["boxes"]:
			patch.set_facecolor("steelblue")
			patch.set_alpha(0.7)

		ax.set_xticks(positions)
		ax.set_xticklabels([str(r["n_agents"]) for r in records])
		ax.set_xlabel("Number of Agents", fontsize=13)
		ax.set_ylabel("D_overhead (ms)", fontsize=13)
		ax.set_title(
			"FlowGentic Coordination Overhead per Invocation\n"
			"(D_overhead = D_resolve + D_collect, whiskers = p5/p95)",
			fontsize=13,
		)
		ax.grid(True, alpha=0.3, axis="y")

		plt.tight_layout()
		if self.plots_dir:
			path = self.plots_dir / "d_overhead_boxplot.png"
			fig.savefig(path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved: {path}")
			try:
				self.discord_notifier.send_discord_notification(
					msg="📊 **d_overhead_boxplot.png**", file_path=str(path)
				)
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		plt.close(fig)

	def _plot_overhead_fraction(self, records: List[Dict[str, Any]]) -> None:
		"""D_overhead / D_total fraction — shows FlowGentic's share of end-to-end time."""
		fig, ax = plt.subplots(figsize=(10, 6))

		n_agents = [r["n_agents"] for r in records]
		fractions = [r.get("overhead_fraction_mean", 0) * 100 for r in records]

		ax.plot(n_agents, fractions, marker="o", linewidth=2, markersize=8, color="steelblue")

		ax.set_xscale("log", base=2)
		ax.set_xlabel("Number of Agents", fontsize=13)
		ax.set_ylabel("FlowGentic Overhead Fraction (%)", fontsize=13)
		ax.set_title(
			"FlowGentic Overhead as Fraction of D_total\n"
			"(≈0% means AsyncFlow dispatch dominates, not FlowGentic)",
			fontsize=13,
		)
		ax.grid(True, alpha=0.3)
		ax.set_ylim(bottom=0)

		plt.tight_layout()
		if self.plots_dir:
			path = self.plots_dir / "overhead_fraction.png"
			fig.savefig(path, dpi=150, bbox_inches="tight")
			logger.info(f"Saved: {path}")
			try:
				self.discord_notifier.send_discord_notification(
					msg="📊 **overhead_fraction.png**", file_path=str(path)
				)
			except Exception as e:
				logger.warning(f"Failed to send plot to Discord: {e}")
		plt.close(fig)
