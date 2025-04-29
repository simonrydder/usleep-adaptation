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
        ids: list[int] | None = None,
        seeds: list[int] | None = None,
    ) -> None:
        self.project = get_project()
        raw_runs: pd.DataFrame = self.project.fetch_runs_table(
            progress_bar=False
        ).to_pandas()

        self.raw_runs = pl.from_pandas(raw_runs)
        self.raw_runs = self.raw_runs.filter(~pl.col("sys/failed"))

        self.runs = self.raw_runs.select(
            pl.col("sys/id"),
            pl.col("sys/tags").alias("tags"),
            pl.col("model/config/experiment/dataset").cast(pl.String).alias("dataset"),
            pl.col("model/config/experiment/method").cast(pl.String).alias("method"),
            pl.col("model/config/experiment/id").cast(pl.Int64()).alias("id"),
            pl.col("model/config/experiment/seed").cast(pl.Int64()).alias("seed"),
            pl.col("model/config/experiment/model").cast(pl.String).alias("model"),
        ).filter(pl.col("model").is_not_null())

        if datasets is not None:
            self.runs = self.runs.filter(pl.col("dataset").is_in(datasets))

        if methods is not None:
            self.runs = self.runs.filter(pl.col("method").is_in(methods))

        if ids is not None:
            self.runs = self.runs.filter(pl.col("id").is_in(ids))

        if seeds is not None:
            self.runs = self.runs.filter(pl.col("seed").is_in(seeds))

        self.experiments = (
            self.runs.with_columns(
                pl.col("tags").str.split(",").alias("tag_parts"),
                pl.concat_list(["dataset", "method", "id", "model", "seed"]).alias(
                    "known_parts"
                ),
            )
            .with_columns(
                pl.struct(["tag_parts", "known_parts"])
                .map_elements(
                    lambda row: next(
                        iter(set(row["tag_parts"]) - set(row["known_parts"])), None
                    ),
                    return_dtype=pl.String(),
                )
                .alias("key")
            )
            .select("sys/id", "key")
        )

        self.keys = (
            self.experiments.select("key")
            .unique(keep="any")
            .sort("key")
            .with_row_index()
        )

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
    method_data = get_method_data(run_ids, key, pbar)
    save_method_data(method_data, key)


def download_data(
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    ids: list[int] | None = None,
) -> None:
    iterator = ExperimentIterator(datasets, methods, ids)
    total_runs = sum(len(run_ids) for _, run_ids in iterator)

    with tqdm(total=total_runs, desc="Downloading data") as pbar:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: _download(*args, pbar=pbar), iterator)


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
    download_data(ids=[99])
    # load_data(datasets=["eesm19"], methods=["BitFit"])
