import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.plot.colors import HIGHLIGHT_COLOR
from src.plot.latex_table_formatter import highlight_and_color_cell
from src.plot.methods import ORDER_MAP, sort_dataframe_by_method_order
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_validation_data


def convergence_plotting() -> None:
    df = _get_data()
    create_latex_table(df)
    plot_max_epoch_boxplot(df)

    pass


def create_latex_table(data: pl.DataFrame) -> str:
    avg = data.group_by("method", maintain_order=True).agg(
        pl.col("max_epoch").mean().alias("avg")
    )
    df_agg = data.group_by("method", "dataset", maintain_order=True).agg(
        pl.col("max_epoch").mean().alias("max_epoch_mean"),
    )
    pivot_values = df_agg.pivot(index="method", on="dataset", values="max_epoch_mean")
    pivot_df = pivot_values.join(avg, on="method", how="left")

    df = pivot_df.to_pandas()
    formatted_df = highlight_and_color_cell(df)

    # Convert to LaTeX
    latex_str = formatted_df.to_latex(index=False, escape=False)
    print(latex_str)
    return latex_str


def _get_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue

        val = extract_validation_data(method_data)
        val = val.drop("accuracy", "loss", "f1")
        dfs.append(val)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")

    df = df.group_by(["dataset", "method", "fold"]).agg(
        pl.col("epoch").max().alias("max_epoch")
    )
    df = df.sort("dataset", "fold")
    df = sort_dataframe_by_method_order(df)
    return df


def plot_max_epoch_boxplot(data: pl.DataFrame) -> None:
    data = data.group_by("dataset", "method").agg(pl.col("max_epoch").median())
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07)
    sns.set_theme(style="whitegrid", context="paper")

    ax = sns.boxplot(
        data=data,
        x="method",
        y="max_epoch",
        showfliers=False,
        fill=False,
        color="black",
        order=ORDER_MAP,
    )

    sns.swarmplot(
        data=data,
        x="method",
        y="max_epoch",
        hue="dataset",
        palette=HIGHLIGHT_COLOR,
        alpha=0.7,
        size=5,
        order=ORDER_MAP,
        ax=ax,
    )

    pass


if __name__ == "__main__":
    table = convergence_plotting()
