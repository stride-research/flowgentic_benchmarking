import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data_generation.experiments.base.base_experiment import BaseExperiment
from data_generation.experiments.synthethic_adaptive.main import SynthethicAdaptive
from data_generation.experiments.throughput_saturation.main import ThroughputSaturation
from data_generation.utils.io_utils import DiscordNotifier, IOUtils

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class FlowGenticBenchmarkManager:
	"""Benchmark harness for FlowGentic scaling tests"""

	def __init__(self, config_path: Path = Path("./config.yml")):
		self.io_utils = IOUtils(config_path)
		self.benchmark_config = self.io_utils.benchmark_config
		self.experiments: Dict[str, dict] = {}

	def register_experiment(self, experiment_name: str, experiment_class):
		data_dir, plots_dir = self.io_utils.create_experiment_directory(
			experiment_name=experiment_name
		)
		self.experiments[experiment_name] = {
			"experiment_class": experiment_class,
			"data_dir": data_dir,
			"plots_dir": plots_dir,
		}
		return data_dir, plots_dir

	def _make_instance(self, experiment_name: str) -> BaseExperiment:
		meta = self.experiments[experiment_name]
		return meta["experiment_class"](
			self.benchmark_config,
			meta["data_dir"],
			meta["plots_dir"],
		)

	async def run_registered_experiments(
		self,
		experiment_name: Optional[str] = None,
		sweep_index: Optional[int] = None,
	):
		"""Run experiments. Optionally filter to a single experiment and/or sweep point."""
		targets = (
			{experiment_name: self.experiments[experiment_name]}
			if experiment_name
			else self.experiments
		)

		for name, _ in targets.items():
			started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			sweep_note = f" (sweep-index={sweep_index})" if sweep_index is not None else ""
			DiscordNotifier().send_discord_notification(
				msg=(
					f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
					f"🚀 **Starting:** `{name}`{sweep_note} at `{started_at}`\n"
					f"**engine:** `{self.benchmark_config.engine_id}`"
				)
			)
			instance = self._make_instance(name)
			await instance.run_experiment(index=sweep_index)

	def finalize_experiments(self, experiment_name: Optional[str] = None):
		"""Generate plots from existing data.jsonl without re-running."""
		targets = (
			{experiment_name: self.experiments[experiment_name]}
			if experiment_name
			else self.experiments
		)
		for name, _ in targets.items():
			self._make_instance(name).finalize()


async def main():
	parser = argparse.ArgumentParser(description="FlowGentic benchmark runner")
	parser.add_argument(
		"--experiment",
		type=str,
		default=None,
		help="Run a specific experiment by name (default: all)",
	)
	parser.add_argument(
		"--sweep-index",
		type=int,
		default=None,
		dest="sweep_index",
		help="Run only the Nth sweep point within the experiment (dragon mode: one call per point)",
	)
	parser.add_argument(
		"--plots-only",
		action="store_true",
		help="Skip execution and only regenerate plots from existing data",
	)
	args = parser.parse_args()

	benchmark = FlowGenticBenchmarkManager()
	benchmark.register_experiment("throughput_saturation", ThroughputSaturation)
	benchmark.register_experiment("syntethic_adaptive", SynthethicAdaptive)

	if args.plots_only:
		benchmark.finalize_experiments(experiment_name=args.experiment)
	else:
		await benchmark.run_registered_experiments(
			experiment_name=args.experiment,
			sweep_index=args.sweep_index,
		)


if __name__ == "__main__":
	asyncio.run(main())
