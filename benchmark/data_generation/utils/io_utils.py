from pathlib import Path
import shutil
import yaml

from data_generation.utils.schemas import BenchmarkConfig

from dotenv import load_dotenv
import os
import requests

load_dotenv()


class IOUtils:
	def __init__(self, config_path: Path = Path("./config.yml")) -> None:
		self.config_path = config_path
		self.benchmark_config = self.get_run_config()
		self._create_core_directories(self.benchmark_config.run_name)

	def _load_config(self, config_path):
		with open(config_path, "r") as file:
			return yaml.safe_load(file)

	def get_run_config(self):
		"""Parses the top-level generic fields from config.yml."""
		config_yaml = self._load_config(self.config_path)
		return BenchmarkConfig(
			run_name=config_yaml["run_name"],
			run_description=config_yaml["run_description"],
			workload_id=config_yaml["workload_id"],
			engine_id=config_yaml.get("engine_id", "asyncflow_local"),
		)

	def _create_core_directories(self, run_configuration_name: str):
		"""Create output directories and initialize analyser

		Ideal map:
			- results
				- {run_configuration_name}
					- {experiment_name}
						- data (json stuff)
							- temp.json
							- foo.sjon
						- plots (the actual plots)
					- config
						- config.yml
		"""
		# Define the paths
		output_dir = Path(f"./results/{run_configuration_name}")
		self.output_dir = output_dir
		config_dir = output_dir / "config"

		# Create folders
		output_dir.mkdir(parents=True, exist_ok=True)
		config_dir.mkdir(parents=True, exist_ok=True)
		shutil.copy(
			self.config_path, config_dir / "config.yml"
		)  # copy config for reproducibility

	def create_experiment_directory(self, experiment_name: str):
		experiment_dir = self.output_dir / "experiments" / experiment_name
		data_dir = experiment_dir / "data"
		plots_dir = experiment_dir / "plots"

		experiment_dir.mkdir(parents=True, exist_ok=True)
		data_dir.mkdir(parents=True, exist_ok=True)
		plots_dir.mkdir(parents=True, exist_ok=True)

		return data_dir, plots_dir

class DiscordNotifier:
	def __init__(self):
		self.webhook_url = os.getenv("DISCORD_WEBHOOK")
	
	def send_discord_notification(self, msg: str, image_path: str = None):
		if not self.webhook_url:
			return None
		
		payload = {"content": msg}
		
		if image_path and os.path.exists(image_path):
			with open(image_path, "rb") as f:
				files = {
					"file": (os.path.basename(image_path), f, "image/png")
				}
				# Send both data (text) and files (image)
				response = requests.post(self.webhook_url, data=payload, files=files)
		else:
			response = requests.post(self.webhook_url, json=payload)
		
		return response