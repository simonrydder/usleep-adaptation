import os
import sys
from pathlib import Path

import yaml

from src.config.experiment import Experiment
from src.experiment_runner import run_experiment


def main():
    try:
        str_id = os.getenv("SLURM_ARRAY_TASK_ID")
        assert isinstance(str_id, str)
        slurm_id = int(str_id)
    except Exception:
        print("SLURM_ARRAY_TASK_ID not found or invalid. Exiting.")
        sys.exit(1)

    config_folder = Path("src/config/yaml/experiments")  # <- update this!
    config_files = sorted([f for f in config_folder.glob("*.yaml") if f.is_file()])

    try:
        config_file = config_files[slurm_id]
    except IndexError:
        print(
            f"SLURM ID {slurm_id} out of range. Only {len(config_files)} config files."
        )
        sys.exit(1)

    print(f"Running job for config file: {config_file.name}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    experiment = Experiment(**config)
    try:
        run_experiment(experiment)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
