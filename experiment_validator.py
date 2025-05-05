import os

import polars as pl

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
        pl.col("model/config/data/dataset").alias("dataset"),
        pl.col("model/config/data/sizes/train")
        .cast(pl.Int32())
        .alias("train_size_records"),
        pl.col("model/config/experiment/train_size")
        .cast(pl.Int32())
        .alias("train_size_subjects"),
        pl.col("model/config/experiment/key").alias("key"),
        pl.col("model/config/experiment/method").alias("method"),
        pl.col("model/config/experiment/seed").cast(pl.Int32()).alias("seed"),
    )
    return transformed


def validate_experiments() -> None:
    df = _get_runs_table()
    completed = df.filter(pl.col("completed"))
    count = (
        completed.group_by(["dataset", "train_size_records", "method"])
        .agg(pl.len())
        .sort("dataset", "method")
    )

    pivot_df = count.pivot(
        on="dataset", index=["train_size_records", "method"], values="len"
    ).sort("train_size_records")
    id_dfs = [grp for _, grp in pivot_df.group_by("train_size_records")]
    pl.Config.set_tbl_rows(-1)
    for id_df in id_dfs:
        print(id_df)
        print()

    sum_ = completed.group_by(["dataset"]).agg(
        pl.len().alias("total"),
    )
    print(sum_.sort("dataset"))
    print("Total runs:", sum_.sum().item(0, "total"))
    print()
    pl.Config.set_tbl_rows(None)


def delete_experiments() -> None:
    df = _get_runs_table()
    df = df.filter(pl.col("completed"))
    filenames = (
        df.with_columns(
            pl.concat_str(
                pl.col("dataset"),
                pl.col("method"),
                pl.col("fold"),
                pl.col("key"),
                separator="-",
            ).alias("filename")
        )
        .get_column("filename")
        .drop_nulls()
        .to_list()
    )

    experiment_names = {f"{file}.yaml" for file in filenames}
    exp_folder = os.path.join("src", "config", "yaml", "experiments")
    experiment_files = set(os.listdir(exp_folder))

    completed = experiment_files & experiment_names

    for file in completed:
        os.remove(os.path.join(exp_folder, file))

    pass


if __name__ == "__main__":
    validate_experiments()
    delete_experiments()
