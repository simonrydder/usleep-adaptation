import os
from concurrent.futures import ThreadPoolExecutor

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
        seeds: list[int] | None = None,
        folds: list[int] | None = None,
        keys: list[str] | None = None,
        train_sizes: list[int | None] | None = None,
        reprocess: bool = False,
    ) -> None:
        self._define_runs()
        self._apply_filters(datasets, methods, seeds, folds, keys, train_sizes)

        if not reprocess:
            self._remove_processed()

        self.experiments = self.runs

        self.keys = (
            self.experiments.select("key")
            .unique(keep="any")
            .sort("key")
            .with_row_index()
        )

    def _define_runs(self) -> None:
        self.project = get_project()
        raw_runs: pd.DataFrame = self.project.fetch_runs_table(
            progress_bar=False
        ).to_pandas()

        self.raw_runs = pl.from_pandas(raw_runs)
        self.raw_runs = self.raw_runs.filter(~pl.col("sys/failed"))

        self.runs = self.raw_runs.select(
            pl.col("sys/id"),
            pl.col("model/config/experiment/key").cast(pl.String()).alias("key"),
            pl.col("model/config/experiment/dataset").cast(pl.String).alias("dataset"),
            pl.col("model/config/experiment/method").cast(pl.String).alias("method"),
            pl.col("model/config/experiment/seed").cast(pl.Int64()).alias("seed"),
            pl.col("fold").cast(pl.Int16()).alias("fold"),
            pl.col("completed").cast(pl.Boolean()),
        )

        if "model/config/experiment/train_size" in self.raw_runs.columns:
            train_size = self.raw_runs.select(
                pl.col("model/config/experiment/train_size")
                .cast(pl.Int64())
                .alias("train_size")
            )
            self.runs = self.runs.with_columns(train_size)
        else:
            self.runs = self.runs.with_columns(
                pl.lit(None).cast(pl.Int16()).alias("train_size")
            )

    def _apply_filters(
        self,
        datasets: list[str] | None = None,
        methods: list[str] | None = None,
        seeds: list[int] | None = None,
        folds: list[int] | None = None,
        keys: list[str] | None = None,
        train_sizes: list[int | None] | None = None,
    ) -> None:
        self.runs = self.runs.filter(pl.col("completed"))

        if datasets is not None:
            self.runs = self.runs.filter(pl.col("dataset").is_in(datasets))

        if methods is not None:
            self.runs = self.runs.filter(pl.col("method").is_in(methods))

        if keys is not None:
            self.runs = self.runs.filter(pl.col("key").is_in(keys))

        if seeds is not None:
            self.runs = self.runs.filter(pl.col("seed").is_in(seeds))

        if folds is not None:
            self.runs = self.runs.filter(pl.col("fold").is_in(folds))

        if train_sizes is not None:
            self.runs = self.runs.filter(
                pl.col("train_size").is_in(train_sizes, nulls_equal=True)
            )

    def _remove_processed(self) -> None:
        downloaded_keys = []
        for _, _, filenames in os.walk("results"):
            for filename in filenames:
                key = filename.split("_")[1].split(".")[0]
                downloaded_keys.append(os.path.join(key))

        self.runs = self.runs.filter(~pl.col("key").is_in(downloaded_keys))

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> "ExperimentIterator":
        self.index = 0
        return self

    def __next__(self) -> tuple[str, list[str]]:
        if self.index >= len(self):
            raise StopIteration

        key = self.keys.row(self.index, named=True)["key"]
        runs_ids = self.experiments.filter(pl.col("key") == key)
        runs_ids = runs_ids.get_column("sys/id").to_list()
        self.index += 1

        return key, runs_ids


def _download(key: str, run_ids: list[str], pbar: tqdm | None) -> None:
    method_data = get_method_data(run_ids, pbar)
    save_method_data(method_data, key)


def download_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    folds: list[int] | None = None,
    keys: list[str] | None = None,
    train_sizes: list[int | None] | None = None,
    reprocess: bool = False,
) -> None:
    iterator = ExperimentIterator(
        datasets, methods, seeds, folds, keys, train_sizes, reprocess
    )
    total_runs = sum(len(run_ids) for _, run_ids in iterator)

    with tqdm(total=total_runs, desc="Downloading data") as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda args: _download(*args, pbar=pbar), iterator)


def load_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[MethodData]:
    data = []

    root = "results"
    for dataset in os.listdir(root):
        folder = os.path.join(root, dataset)

        if not os.path.isdir(folder):
            continue

        if datasets is not None and dataset not in datasets:
            continue

        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)

            if not os.path.isfile(file):
                continue

            method = filename.split("_")[0]

            if methods is not None and method not in methods:
                continue

            data.append(load_method_data(dataset, filename))

    return data


if __name__ == "__main__":
    download_data(datasets=["dod_h", "eesm23"])
    # load_data()
