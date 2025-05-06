import os
import sys
from contextlib import redirect_stderr, redirect_stdout

from src.utils.experiment_reader import (
    NoPendingException,
    load_and_update_pending_with_lock,
    update_running_with_lock,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from src.config.experiment import get_experiment_name
from src.experiment_runner import run_experiment


def main():
    try:
        dataset = os.getenv("DATASET")
        assert isinstance(dataset, str)
    except Exception:
        print("DATASET environment variable not found. Exiting.")
        sys.exit(1)

    config_folder = os.path.join("src", "config", "yaml", "experiments")
    experiments_csv = os.path.join(config_folder, f"{dataset}.csv")

    for _ in range(1000):
        try:
            index, experiment = load_and_update_pending_with_lock(experiments_csv)
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
