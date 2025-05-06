from time import sleep
from typing import Any

import polars as pl
from filelock import FileLock

from src.config.experiment import Experiment

CSV_FILE = "experiments.csv"
LOCK_FILE = CSV_FILE + ".lock"


class NoPendingException(Exception):
    pass


def _load_experiments(csv_path: str) -> pl.DataFrame:
    return pl.read_csv(csv_path)


def _save_experiments(df: pl.DataFrame, csv_path: str) -> None:
    df.write_csv(csv_path)


def _get_first_pending_row(df: pl.DataFrame) -> dict[str, Any]:
    pending = df.filter(pl.col("status") == "pending")

    if pending.is_empty():
        raise NoPendingException

    return pending.row(0, named=True)


def _update_dataframe_row(df: pl.DataFrame, new_row: dict[str, Any]) -> pl.DataFrame:
    row_index = new_row["index"]
    return pl.concat(
        [
            df.slice(0, row_index),
            pl.DataFrame([new_row], schema=df.schema),
            df.slice(row_index + 1, df.height),
        ]
    )


def _update_row_status(row: dict[str, Any], new_status: str) -> dict[str, Any]:
    row["status"] = new_status
    return row


def _create_row(index: int, experiment: Experiment) -> dict[str, Any]:
    row = experiment.model_dump()
    row["index"] = index

    return row


def load_and_update_pending(csv_path: str) -> tuple[int, Experiment]:
    df = _load_experiments(csv_path)
    next_pending = _get_first_pending_row(df)

    index = next_pending["index"]
    experiment = Experiment(**next_pending)

    next_running = _update_row_status(next_pending, "running")
    updated_df = _update_dataframe_row(df, next_running)
    _save_experiments(updated_df, csv_path)

    return index, experiment


def update_running(
    csv_path: str, index: int, experiment: Experiment, new_status: str
) -> None:
    df = _load_experiments(csv_path)

    running_row = _create_row(index, experiment)
    complete_row = _update_row_status(running_row, new_status)

    updated_df = _update_dataframe_row(df, complete_row)
    _save_experiments(updated_df, csv_path)


def load_and_update_pending_with_lock(csv_path: str) -> tuple[int, Experiment]:
    lock_path = csv_path + ".lock"

    with FileLock(lock_path, timeout=10):
        index, exp = load_and_update_pending(csv_path=csv_path)

    return index, exp


def update_running_with_lock(
    csv_path: str, index: int, experiment: Experiment, new_status: str
) -> None:
    lock_path = csv_path + ".lock"

    with FileLock(lock_path, timeout=10):
        update_running(
            csv_path=csv_path, index=index, experiment=experiment, new_status=new_status
        )

    return None


if __name__ == "__main__":
    csv_path = "experiments.csv"
    for _ in range(1000):
        try:
            index, exp = load_and_update_pending_with_lock(csv_path)
        except NoPendingException:
            break

        print(f"Running experiment {index}: {exp}")
        sleep(0.1)

        update_running_with_lock(csv_path, index, exp, "done")
