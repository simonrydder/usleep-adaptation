import os

import pandas as pd
import polars as pl
from tqdm import tqdm

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


def get_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    ids: list[int] | None = None,
) -> list[MethodData]:
    data = []
    for run_ids in tqdm(
        ExperimentIterator(datasets, methods, ids), desc="Downloading experiments"
    ):
        method_data = get_method_data(run_ids)
        save_method_data(method_data)

        data.append(method_data)

    return data


def load_data() -> list[MethodData]:
    data = []

    root = os.path.join("results")
    for dataset in os.listdir(root):
        dataset_folder = os.path.join(root, dataset)

        for id in os.listdir(dataset_folder):
            id_folder = os.path.join(dataset_folder, id)

            for method in os.listdir(id_folder):
                data.append(load_method_data(dataset, id, method))

    return data


if __name__ == "__main__":
    get_data()
    x = load_data()
    pass
