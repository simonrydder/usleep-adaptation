import os
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from neptune import Run
from tqdm import tqdm

from src.utils.neptune_api.neptune_api import get_data_scalar, get_project, get_run


def download_checkpoint(run: Run) -> None:
    """
    Download the checkpoint from the given run.

    Args:
        run (Run): The Neptune run object.
    """
    dest = get_checkpoint_path(run)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    run["training/model/checkpoints/best"].download(
        destination=dest, progress_bar=False
    )


def get_checkpoint_path(run: Run) -> str:
    dataset = get_data_scalar(run, "model/config/data/dataset")
    method = get_data_scalar(run, "model/config/experiment/method")
    model = get_data_scalar(run, "model/config/experiment/model")
    id = get_data_scalar(run, "model/config/experiment/id")
    tags = get_data_scalar(run, "sys/tags")
    fold = get_data_scalar(run, "fold")
    key = next(iter((tags - {dataset, method, model, str(id)})))

    return combine_checkpoint_path(dataset, method, key, str(fold))


def combine_checkpoint_path(dataset: str, method: str, key: str, fold: str) -> str:
    return os.path.join("ckpt_results", dataset, method, f"{key}_{fold}.ckpt")


def _download_checkpoint_for_run(run_row: dict) -> None:
    id = run_row["sys/id"]
    exp_id = run_row["model/config/experiment/id"]

    if exp_id in [42, 43, 44, 45, 46, 99]:
        return None

    run = get_run(id)

    path = get_checkpoint_path(run)
    if os.path.exists(path):
        return None

    download_checkpoint(run)


def download_missing_checkpoints() -> None:
    project = get_project()
    runs = pl.from_pandas(project.fetch_runs_table().to_pandas())

    run_rows = list(runs.iter_rows(named=True))  # evaluate into list once

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(_download_checkpoint_for_run, run_rows),
                total=len(run_rows),
                desc="Downloading checkpoints",
            )
        )


if __name__ == "__main__":
    download_missing_checkpoints()
