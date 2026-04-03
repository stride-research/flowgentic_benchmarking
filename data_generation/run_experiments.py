import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
			self.io_utils.config_path,
		)

	async def run_registered_experiments(
		self,
		experiment_name: str,
		sweep_index: int,
		iter_index: int,
	):
		"""Run a single sweep point of a single experiment."""
		cfg = self.benchmark_config
		started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		DiscordNotifier().send_discord_notification(
			msg=(
				f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
				f"🚀 **Starting** `{experiment_name}`\n"
				f"```\n"
				f"run      : {cfg.run_name}\n"
				f"engine   : {cfg.engine_id}\n"
				f"sweep    : {sweep_index}\n"
				f"iter     : {iter_index}\n"
				f"started  : {started_at}\n"
				f"```"
			)
		)
		instance = self._make_instance(experiment_name)
		await instance.run_experiment(sweep_index=sweep_index, iter_index=iter_index)

	def finalize_experiments(self, experiment_name: str = None):
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
		"--config",
		type=Path,
		default=Path("config.yml"),
		dest="config_path",
		help="Path to config.yml (default: config.yml in CWD)",
	)
	parser.add_argument(
		"--experiment",
		type=str,
		default=None,
		help="Experiment name (required unless --plots-only)",
	)
	parser.add_argument(
		"--sweep-index",
		type=int,
		default=None,
		dest="sweep_index",
		help="Index into the sweep list",
	)
	parser.add_argument(
		"--iter",
		type=int,
		default=0,
		dest="iter_index",
		help="Repetition index",
	)
	parser.add_argument(
		"--plots-only",
		action="store_true",
		help="Skip execution and only regenerate plots from existing data",
	)
	args = parser.parse_args()

	benchmark = FlowGenticBenchmarkManager(config_path=args.config_path)
	benchmark.register_experiment("throughput_saturation", ThroughputSaturation)
	benchmark.register_experiment("syntethic_adaptive", SynthethicAdaptive)

	if args.plots_only:
		benchmark.finalize_experiments(experiment_name=args.experiment)
	else:
		if args.experiment is None or args.sweep_index is None:
			parser.error("--experiment, --sweep-index are all required (use --plots-only to regenerate plots)")
		await benchmark.run_registered_experiments(
			experiment_name=args.experiment,
			sweep_index=args.sweep_index,
			iter_index=args.iter_index,
		)


if __name__ == "__main__":
	asyncio.run(main())
