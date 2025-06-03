import polars as pl

from src.plot.latex_table_formatter import delta_kappa_format
from src.plot.methods import sort_dataframe_by_method_order
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance, extract_settings


def delta_kappa_table() -> str:
    df = _get_data()
    avg = df.group_by("method", maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("avg")
    )
    df_agg = df.group_by(["dataset", "method"], maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("delta_kappa_mean"),
    )

    pivot_values = df_agg.pivot(index="method", on="dataset", values="delta_kappa_mean")
    pivot_df = pivot_values.join(avg, on="method", how="left").sort(
        "avg", descending=True
    )

    pl.Config.set_tbl_rows(25)
    pl.Config.set_tbl_cols(15)
    print(pivot_df)
    formatted_df = delta_kappa_format(pivot_df.to_pandas())

    return formatted_df.to_latex(index=False, escape=False)


def parameter_scaled_delta_kappa_table() -> str:
    df = _get_data()
    avg = df.group_by("method", maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("avg")
    )
    df_agg = df.group_by(["dataset", "method"], maintain_order=True).agg(
        pl.col("delta_kappa").mean().alias("delta_kappa_mean"),
    )

    pivot_values = df_agg.pivot(index="method", on="dataset", values="delta_kappa_mean")
    pivot_df = pivot_values.join(avg, on="method", how="left").sort(
        "avg", descending=True
    )

    pl.Config.set_tbl_rows(25)
    pl.Config.set_tbl_cols(15)
    print(pivot_df)

    return ""


def _get_data() -> pl.DataFrame:
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
    print(parameter_scaled_delta_kappa_table())
    print(delta_kappa_table())
