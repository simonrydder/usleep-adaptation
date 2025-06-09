import polars as pl

from src.plot.latex_table_formatter import delta_kappa_format
from src.plot.methods import sort_dataframe_by_method_order
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance, extract_settings


def delta_kappa_table() -> None:
    df = _get_data()

    full_train_size = df.sort("dataset", "method", "record", "train_size").unique(
        ["dataset", "method", "record"], keep="last", maintain_order=True
    )

    min_train_size = df.sort("train_size").unique(
        ["dataset", "method", "record"], keep="first", maintain_order=True
    )

    print(get_delta_kappa_latex_table(sort_dataframe_by_method_order(full_train_size)))
    print(
        full_train_size.group_by("dataset", maintain_order=True)
        .agg(
            pl.col("delta_kappa").mean().alias("mean") * 100,
            pl.col("delta_kappa").std().alias("std") * 100,
        )
        .to_pandas()
        .T.to_latex(index=False, escape=False)
    )
    print()
    print(get_delta_kappa_latex_table(sort_dataframe_by_method_order(min_train_size)))
    print(
        min_train_size.group_by("dataset", maintain_order=True)
        .agg(
            pl.col("delta_kappa").mean().alias("mean") * 100,
            pl.col("delta_kappa").std().alias("std") * 100,
        )
        .to_pandas()
        .T.to_latex(index=False, escape=False)
    )
    pass


def get_delta_kappa_latex_table(df: pl.DataFrame):
    avg = df.group_by("method", maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("avg"),
        pl.col("delta_kappa").std().alias("std"),
    )
    df_agg = df.group_by(["dataset", "method"], maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("delta_kappa_mean"),
    )

    pivot_values = df_agg.pivot(index="method", on="dataset", values="delta_kappa_mean")
    pivot_df = pivot_values.join(avg, on="method", how="left").sort(
        "avg", descending=True
    )

    # pl.Config.set_tbl_rows(25)
    # pl.Config.set_tbl_cols(15)
    # print(pivot_df)
    formatted_df = delta_kappa_format(pivot_df.to_pandas())

    return formatted_df.to_latex(index=False, escape=False)


def _get_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        test = extract_performance(method_data, "new").select(
            "dataset", "method", "record", "key", "kappa"
        )
        base = extract_performance(method_data, "org").select(
            "dataset", "method", "record", "key", pl.col("kappa").alias("base_kappa")
        )
        delta_kappa = test.join(
            base, on=["dataset", "method", "record", "key"], how="left"
        ).with_columns((pl.col("kappa") - pl.col("base_kappa")).alias("delta_kappa"))

        settings = (
            extract_settings(method_data)
            .drop("fold")
            .sort("train_size")
            .unique(["key"], keep="first")
        )
        delta_kappa = delta_kappa.join(
            settings, on=["dataset", "method", "key"], how="left"
        )
        dfs.append(delta_kappa.drop("key"))

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    df = sort_dataframe_by_method_order(df)
    return df


def _old_get_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue

        test = extract_performance(method_data, "new").select(
            "dataset", "method", "record", "kappa"
        )
        base = extract_performance(method_data, "org").select(
            "dataset", "method", "record", pl.col("kappa").alias("base_kappa")
        )
        delta_kappa = test.join(
            base, on=["dataset", "method", "record"], how="left"
        ).with_columns((pl.col("kappa") - pl.col("base_kappa")).alias("delta_kappa"))

        free_params = extract_settings(method_data).select(
            "dataset", "method", "free_parameters"
        )
        delta_kappa = delta_kappa.join(
            free_params, on=["dataset", "method"], how="left"
        )
        dfs.append(delta_kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    df = sort_dataframe_by_method_order(df)
    return df


if __name__ == "__main__":
    delta_kappa_table()
