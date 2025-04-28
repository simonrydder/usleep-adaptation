import polars as pl

from src.utils.neptune_api.neptune_api import get_project


def _get_runs_table() -> pl.DataFrame:
    project = get_project()
    runs_table = project.fetch_runs_table()
    df = pl.from_pandas(runs_table.to_pandas())
    transformed = df.select(
        pl.col("sys/failed").alias("failed"),
        pl.col("sys/tags").alias("tags"),
        pl.col("sys/id"),
        pl.col("fold"),
        # pl.col("model/config/adapter/method").alias("method"),
        pl.col("model/config/data/dataset").alias("dataset"),
        pl.col("model/config/data/sizes/train").alias("train_size_records"),
        pl.col("model/config/data/train_size").alias("train_size_subjects"),
        pl.col("model/config/experiment/id").alias("id"),
        pl.col("model/config/experiment/method").alias("method"),
    )
    return transformed


def validate_experiments() -> None:
    df = _get_runs_table()
    df = df.filter(pl.col("dataset").is_not_null())
    count = (
        df.filter(~pl.col("failed"))
        .group_by(["dataset", "method", "id"])
        .agg(pl.len())
        .sort("dataset", "id", "method")
    )
    pivot_df = count.pivot(on="dataset", index=["method", "id"], values="len")
    id_dfs = [grp for _, grp in pivot_df.group_by("id")]
    pl.Config.set_tbl_rows(-1)
    for id_df in id_dfs:
        print(id_df)
        print()

    sum_ = (
        count.drop("method")
        .group_by(["dataset", "id"])
        .agg(pl.sum("len").alias("total"))
    ).with_columns((pl.col("total") / 22).alias("folds"))
    print(sum_.sort("dataset", "id"))
    print("Total runs:", sum_.sum().item(0, "total"))
    print()
    pl.Config.set_tbl_rows(None)


def delete_runs() -> None:
    pass


if __name__ == "__main__":
    validate_experiments()
