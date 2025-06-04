import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
from contextlib import redirect_stderr, redirect_stdout

from src.config.experiment import get_experiment_name
from src.experiment_runner import run_experiment
from src.utils.experiment_reader import (
    NoPendingException,
    load_and_update_pending_with_lock,
    update_running_with_lock,
)


def main():
    try:
        dataset = os.getenv("DATASET")
        assert isinstance(dataset, str)
        print(f'Selected dataset {dataset}')
    except Exception:
        print("DATASET environment variable not found. Exiting.")
        sys.exit(1)

    try:
        max_runs = os.getenv("MAX_RUNS")
        assert isinstance(max_runs, str)
        max_runs = int(max_runs)
    except Exception:
        print("MAX_RUNS environment variable not found. Setting max_runs to 1.")
        max_runs = 1

    config_folder = os.path.join("src", "config", "yaml", "experiments")
    experiments_csv = os.path.join(config_folder, f"{dataset}.csv")

    for _ in range(max_runs):
        try:
            index, experiment = load_and_update_pending_with_lock(experiments_csv)
            print(f"Try run experiment: {get_experiment_name(experiment)}")
        except NoPendingException:
            print(f"No more pending experiments for {dataset}. Exiting.")
            sys.exit(0)

        log_dir = os.path.join("logs", dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{get_experiment_name(experiment)}.log")

        with open(log_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
            try:
                print(f"Running experiment index: {index}")
                run_experiment(experiment)
                update_running_with_lock(experiments_csv, index, experiment, "done")
            except Exception as e:
                print("Error during experiment. Resetting to pending.")
                update_running_with_lock(experiments_csv, index, experiment, "pending")
                print(e)


if __name__ == "__main__":
    main()
