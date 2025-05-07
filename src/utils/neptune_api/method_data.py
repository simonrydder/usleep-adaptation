import json
import os
from threading import Lock
from typing import Literal

import polars as pl
from pydantic import BaseModel
from tqdm import tqdm

from src.config.experiment import Experiment
from src.utils.neptune_api.fold_data import (
    FoldData,
    extract_fold_settings,
    get_fold_data,
)
from src.utils.neptune_api.neptune_api import get_data_scalar, get_run
from src.utils.neptune_api.performance_data import PerformanceData, get_performance_data

tqdm_lock = Lock()


class MethodData(BaseModel):
    key: str
    method: str
    dataset: str
    model: str
    seed: int
    train_size: int | None = None
    original_performance: list[PerformanceData]
    new_performance: list[PerformanceData]
    folds: dict[int, FoldData]


def get_method_data(run_ids: list[str], pbar: tqdm | None = None) -> MethodData:
    """Gets the method data for a list of run_ids. These MUST be ids of the different folds for the same experiment."""
    folds = {}
    org_performance = []
    new_performance = []
    experiment = None
    for id in run_ids:
        run = get_run(id)
        fold_data = get_fold_data(run)
        fold_id = get_data_scalar(run, "fold")
        folds.update({fold_id: fold_data})

        org_performance += get_performance_data(run, "org", "test")
        new_performance += get_performance_data(run, "new", "test")
        experiment = Experiment(**get_data_scalar(run, "model/config/experiment"))

        if pbar is not None:
            with tqdm_lock:
                pbar.update(1)

    assert experiment is not None

    return MethodData(
        key=experiment.key,
        method=experiment.method,
        dataset=experiment.dataset,
        model=experiment.model,
        seed=experiment.seed,
        train_size=experiment.train_size,
        original_performance=org_performance,
        new_performance=new_performance,
        folds=folds,
    )


def save_method_data(data: MethodData, key: str) -> None:
    filename = "_".join([data.method, key])
    folder = os.path.join("results", data.dataset)
    file = os.path.join(folder, f"{filename}.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(file, "w") as f:
        json.dump(data.model_dump(), f, indent=4)


def load_method_data(dataset: str, method_file: str) -> MethodData:
    file = os.path.join("results", dataset, method_file)
    with open(file, "r") as f:
        data = json.load(f)

    return MethodData(**data)


def _get_performance_list(
    data: MethodData, mode: Literal["new", "org"]
) -> list[PerformanceData]:
    match mode:
        case "new":
            return data.new_performance
        case "org":
            return data.original_performance
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def _add_index_columns(df: pl.DataFrame, data: MethodData) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(data.key).alias("key"),
        pl.lit(data.dataset).alias("dataset"),
        pl.lit(data.method).alias("method"),
        pl.lit(data.seed).alias("seed"),
    )


def extract_performance(data: MethodData, mode: Literal["new", "org"]) -> pl.DataFrame:
    perf = _get_performance_list(data, mode)
    df = pl.DataFrame(perf)

    return _add_index_columns(df, data)


def extract_settings(data: MethodData) -> pl.DataFrame:
    dfs = []
    for fold, fold_data in data.folds.items():
        fold_setting = extract_fold_settings(fold_data, fold)
        df = _add_index_columns(fold_setting, data)
        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def extract_validation_data(data: MethodData, train: bool = False) -> pl.DataFrame:
    dfs = []
    for fold, fold_data in data.folds.items():
        step_data = fold_data.train_step if train else fold_data.validation_step

        validation = pl.DataFrame(step_data)
        validation = _add_index_columns(validation, data)
        validation = (
            validation.with_columns(pl.lit(fold).alias("fold"))
            .with_row_index("step")
            .with_columns(
                (pl.col("step") // fold_data.config.data.sizes.validation).alias(
                    "epoch"
                )
            )
        )
        dfs.append(validation)

    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":
    data = get_method_data(["US-3490"])
    save_method_data(data, "test")
    loaded = load_method_data("eesm19", "LoRA10_test.json")
    pass
