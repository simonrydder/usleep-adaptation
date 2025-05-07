import os

import polars as pl

from experiment_setup_new import EXPERIMENT_SCHEMA
from src.config.experiment import EXPERIMENTS_FOLDER
from src.utils.neptune_api.neptune_api import get_project


def _get_runs_table() -> pl.DataFrame:
    project = get_project()
    runs_table = project.fetch_runs_table()
    df = pl.from_pandas(runs_table.to_pandas())
    transformed = df.select(
        pl.col("sys/id"),
        pl.col("sys/failed").alias("failed"),
        pl.col("completed"),
        pl.col("fold"),
        pl.col("model/config/experiment/dataset").alias("dataset"),
        pl.col("model/config/data/sizes/train")
        .cast(pl.Int32())
        .alias("train_size_records"),
        pl.col("model/config/experiment/train_size")
        .cast(pl.Int32())
        .alias("train_size"),
        pl.col("model/config/experiment/key").alias("key"),
        pl.col("model/config/experiment/method").alias("method"),
        pl.col("model/config/experiment/model").alias("model"),
        pl.col("model/config/experiment/trainer").alias("trainer"),
        pl.col("model/config/experiment/seed").cast(pl.Int32()).alias("seed"),
    )
    return transformed.with_columns(
        [
            pl.col(col).cast(dtype)
            for col, dtype in EXPERIMENT_SCHEMA.items()
            if col in df.columns
        ]
    )


def _get_experiment_table() -> pl.DataFrame:
    dfs = []
    for file in os.listdir(EXPERIMENTS_FOLDER):
        if not file.endswith(".csv"):
            continue

        df = pl.read_csv(
            os.path.join(EXPERIMENTS_FOLDER, file), schema=EXPERIMENT_SCHEMA
        )
        dfs.append(df)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    return df


def _update_status_to_match_completed(
    experiments: pl.DataFrame, runs: pl.DataFrame
) -> pl.DataFrame:
    join_columns = [
        "key",
        "dataset",
        "method",
        "model",
        "trainer",
        "train_size",
        "fold",
        "seed",
    ]

    return (
        experiments.join(runs, on=join_columns, how="left", nulls_equal=True)
        .with_columns(
            pl.when(pl.col("status_marker").is_not_null())
            .then(pl.lit("done"))
            .otherwise(pl.lit("pending"))
            .alias("status")
        )
        .select(*EXPERIMENT_SCHEMA.keys())
    )


def experiment_validator() -> None:
    df = _get_runs_table()
    completed = df.filter(pl.col("completed")).with_columns(
        pl.lit("done").alias("status_marker")
    )

    experiments = _get_experiment_table()
    updated = _update_status_to_match_completed(experiments, completed)

    grper = updated.group_by("dataset")
    for _, grp in grper:
        dataset = grp.item(0, "dataset")
        grp.write_csv(os.path.join(EXPERIMENTS_FOLDER, f"{dataset}.csv"))

    pending_count = grper.agg(
        (pl.col("status") == "pending").sum().alias("pending_count")
    ).sort("dataset")
    print(pending_count)


def print_duplicates():
    df = _get_runs_table()
    complete = df.filter(pl.col("completed"))

    complete.filter(
        pl.struct(["dataset", "method", "train_size", "seed", "fold"]).is_duplicated()
    )
    pass


if __name__ == "__main__":
    experiment_validator()
    # print_duplicates()
