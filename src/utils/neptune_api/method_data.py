import json
import os
from typing import Literal

import polars as pl
from pydantic import BaseModel
from tqdm import tqdm

from src.utils.neptune_api.fold_data import (
    FoldData,
    extract_fold_settings,
    get_fold_data,
)
from src.utils.neptune_api.neptune_api import get_data_scalar, get_run
from src.utils.neptune_api.performance_data import PerformanceData, get_performance_data


class MethodData(BaseModel):
    original_performance: list[PerformanceData]
    new_performance: list[PerformanceData]
    folds: dict[int, FoldData]


def get_method_data(run_ids: list[str]) -> MethodData:
    folds = {}
    org_performance = []
    new_performance = []
    for id in tqdm(run_ids, desc="Iterating over runs"):
        run = get_run(id)
        fold_data = get_fold_data(run)

        fold_id = get_data_scalar(run, "fold")
        folds.update({fold_id: fold_data})

        org_performance += get_performance_data(run, "org", "test")
        new_performance += get_performance_data(run, "new", "test")

    return MethodData(
        original_performance=org_performance,
        new_performance=new_performance,
        folds=folds,
    )


def save_method_data(method_data: MethodData, dataset: str, method: str) -> None:
    folder = os.path.join("results", dataset)
    file = os.path.join(folder, f"{method}.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(file, "w") as f:
        json.dump(method_data.model_dump(), f, indent=4)


def load_method_data(dataset: str, method: str) -> MethodData:
    file = os.path.join("results", dataset, f"{method}.json")
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
    data: dict[str, MethodData], mode: Literal["new", "org"]
) -> pl.DataFrame:
    dfs = []
    for method, method_data in data.items():
        perf = _get_performance_list(method_data, mode)
        df = pl.DataFrame(perf).with_columns(pl.lit(method).alias("method"))
        dfs.append(df)

    return pl.concat(dfs, how="vertical")


def extract_settings(data: dict[str, MethodData]) -> pl.DataFrame:
    dfs = []
    for method, method_data in data.items():
        for fold, fold_data in method_data.folds.items():
            fold_setting = extract_fold_settings(fold_data, fold)
            df = fold_setting.with_columns(pl.lit(method).alias("method"))
            dfs.append(df)

    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":
    x = get_method_data(
        [
            "US-440",
            "US-431",
            "US-419",
            # "US-411",
            # "US-392",
            # "US-378",
            # "US-371",
            # "US-368",
            # "US-366",
            # "US-351",
        ]
    )
    save_method_data(x, "eesm19", "BitFit")
    y = load_method_data("eesm19", "BitFit")
    pass
