import json
import os
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


class MethodData(BaseModel):
    method: str
    dataset: str
    id: int
    model: str
    original_performance: list[PerformanceData]
    new_performance: list[PerformanceData]
    folds: dict[int, FoldData]


def get_method_data(run_ids: list[str]) -> MethodData:
    """Gets the method data for a list of run_ids. These MUST be ids of the different folds for the same experiment."""
    folds = {}
    org_performance = []
    new_performance = []
    experiment = None
    for id in tqdm(run_ids, desc="Iterating over runs"):
        run = get_run(id)
        fold_data = get_fold_data(run)
        fold_id = get_data_scalar(run, "fold")
        folds.update({fold_id: fold_data})

        org_performance += get_performance_data(run, "org", "test")
        new_performance += get_performance_data(run, "new", "test")
        experiment = Experiment(**get_data_scalar(run, "model/config/experiment"))

    assert experiment is not None

    if isinstance(experiment.id, str):
        experiment.id = int(experiment.id.split("_")[-1])  # type: ignore

    return MethodData(
        method=experiment.method,
        dataset=experiment.dataset,
        id=experiment.id,
        model=experiment.model,
        original_performance=org_performance,
        new_performance=new_performance,
        folds=folds,
    )


def save_method_data(method_data: MethodData) -> None:
    folder = os.path.join("results", method_data.dataset, str(method_data.id))
    file = os.path.join(folder, f"{method_data.method}.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(file, "w") as f:
        json.dump(method_data.model_dump(), f, indent=4)


def load_method_data(dataset: str, id: str | int, method_file: str) -> MethodData:
    file = os.path.join("results", dataset, str(id), method_file)
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


def extract_performance(
    data: list[MethodData], mode: Literal["new", "org"]
) -> pl.DataFrame:
    dfs = []
    for method_data in data:
        perf = _get_performance_list(method_data, mode)
        df = pl.DataFrame(perf).with_columns(
            pl.lit(method_data.method).alias("method"),
            pl.lit(method_data.dataset).alias("dataset"),
        )
        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def extract_settings(data: list[MethodData]) -> pl.DataFrame:
    dfs = []
    for method_data in data:
        for fold, fold_data in method_data.folds.items():
            fold_setting = extract_fold_settings(fold_data, fold)
            df = fold_setting.with_columns(
                pl.lit(method_data.method).alias("method"),
                pl.lit(method_data.dataset).alias("dataset"),
            )
            dfs.append(df)

    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":
    x = get_method_data(
        [
            "US-950",
            "US-949",
        ]
    )
    save_method_data(x)
    y = load_method_data(x.dataset, x.id, x.method)
    pass
