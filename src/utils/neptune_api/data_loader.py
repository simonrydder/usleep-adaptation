import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import polars as pl

from src.utils.neptune_api.method_data import (
    MethodData,
    get_method_data,
    load_method_data,
    save_method_data,
)
from src.utils.neptune_api.neptune_api import get_project


class ExperimentIterator:
    def __init__(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | None = None,
        ids: list[int] | None = None,
    ) -> None:
        self.project = get_project()
        runs: pd.DataFrame = self.project.fetch_runs_table(
            progress_bar=False
        ).to_pandas()

        self.runs = pl.from_pandas(runs)
        self.runs = self.runs.filter(~pl.col("sys/failed"))

        self.dataset_col = "model/config/experiment/dataset"
        if datasets is not None:
            self.runs = self.runs.filter(pl.col(self.dataset_col).is_in(datasets))

        self.method_col = "model/config/experiment/method"
        if methods is not None:
            self.runs = self.runs.filter(pl.col(self.method_col).is_in(methods))

        self.id_col = "model/config/experiment/id"
        if ids is not None:
            self.runs = self.runs.filter(pl.col(self.id_col).is_in(ids))

        self.experiments = (
            self.runs.select([self.dataset_col, self.method_col, self.id_col])
            .unique(keep="any")
            .sort(self.dataset_col, self.id_col, self.method_col)
            .with_row_index()
        )
        print()

    def __len__(self) -> int:
        return len(self.experiments)

    def __iter__(self) -> "ExperimentIterator":
        self.index = 0
        return self

    def __next__(self) -> list[str]:
        if self.index >= len(self):
            raise StopIteration

        experiment = self.experiments.row(self.index, named=True)
        runs_ids = self.runs.filter(
            pl.col(self.dataset_col) == experiment[self.dataset_col],
            pl.col(self.method_col) == experiment[self.method_col],
            pl.col(self.id_col) == experiment[self.id_col],
        )
        runs_ids = runs_ids.get_column("sys/id").to_list()
        self.index += 1

        return runs_ids


def _fetch_and_save(run_ids: list[str]) -> MethodData:
    method_data = get_method_data(run_ids)
    save_method_data(method_data)
    return method_data


def get_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    ids: list[int] | None = None,
) -> list[MethodData]:
    with ThreadPoolExecutor() as executor:
        results = list(
            # tqdm(
            executor.map(_fetch_and_save, ExperimentIterator(datasets, methods, ids)),
            # total=len(ExperimentIterator(datasets, methods, ids)),
            # desc="Downloading experiments",
            # )
        )
    return results


def load_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    ids: list[int] | None = None,
) -> list[MethodData]:
    data = []

    root = "results"
    for folder, _, files in os.walk(root):
        for file in files:
            method, _ = os.path.splitext(file)

            if methods is not None and method not in methods:
                continue

            folders = os.path.normpath(folder).split(os.sep)
            assert len(folders) == 3, f"Unexpected folder structure: {folders}"
            _, dataset, id = folders
            id = int(id)

            if datasets is not None and dataset not in datasets:
                continue

            if ids is not None and id not in ids:
                continue

            data.append(load_method_data(dataset, id, file))

    return data


if __name__ == "__main__":
    get_data(
        ids=[99],
        methods=[
            "Full_3",
            "Full_4",
            "Full_6",
            "LoRA20_3",
            "LoRA20_4",
            # "LoRA20_6",
            "SCL20_3",
            "SCL20_4",
            "SCL20_6",
        ],
    )
    # load_data(datasets=["eesm19"], methods=["BitFit"])
