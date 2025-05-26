import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.plot.colors import HIGHLIGHT_COLOR
from src.utils.figures import adjust_axis_font
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance


def joined_delta_kappa_vs_methods_plot() -> None:
    df = _get_data()
    plot_joined_delta_kappa_vs_methods(df)


def plot_joined_delta_kappa_vs_methods(data: pl.DataFrame) -> None:
    overall_avg = data.group_by("method").agg(pl.col("delta_kappa").mean())
    dataset_avg = data.group_by("dataset", "method").agg(pl.col("delta_kappa").mean())

    overall_avg = overall_avg.sort("delta_kappa")
    dataset_avg = dataset_avg.sort("delta_kappa")

    plt.figure(figsize=(18, 6))
    sns.set_theme(style="whitegrid", context="paper")

    method_order = overall_avg.sort("delta_kappa").get_column("method").to_list()
    ax = sns.boxplot(
        data=data,
        x="method",
        y="delta_kappa",
        # showfliers=False,
        fill=False,
        color="black",
        order=method_order,
        showfliers=False,
    )

    sns.swarmplot(
        data=dataset_avg,
        x="method",
        y="delta_kappa",
        hue="dataset",
        palette=HIGHLIGHT_COLOR,
        alpha=0.7,
        size=5,
        order=method_order,
        ax=ax,
    )

    # ax = sns.violinplot(
    #     data=data,
    #     x="method",
    #     y="delta_kappa",
    #     # showfliers=False,
    #     fill=False,
    #     color="black",
    #     order=method_order,
    #     inner="quart",
    # )

    plt.suptitle("Delta Kappa vs. Method", size=14)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.11)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Kappa", fontsize=12)
    plt.xticks(rotation=20, ha="center")

    adjust_axis_font(ax.xaxis, size=10)
    adjust_axis_font(ax.yaxis, size=10)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(range(len(method_order)))
    secax.set_xticklabels(
        [f"{mean:.3f}" for mean in overall_avg.get_column("delta_kappa")],
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    secax.set_xlabel("Mean Kappa", fontsize=12)


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

        dfs.append(delta_kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    return df.sort("dataset", "method")


if __name__ == "__main__":
    joined_delta_kappa_vs_methods_plot()
