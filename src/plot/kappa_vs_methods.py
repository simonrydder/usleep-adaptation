import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.plot.colors import BASE_COLOR, HIGHLIGHT_COLOR
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance


def plot_kappa_vs_methods(show: bool = False) -> None:
    data = get_kappa_data()

    for _, grp_df in data.group_by("dataset"):
        _plot_kappa_vs_methods(grp_df, show=show)


def get_kappa_data() -> pl.DataFrame:
    raw_data = load_data(ids=[0])

    dfs = []
    for method_data in raw_data:
        test = extract_performance(method_data, "new")
        base = extract_performance(method_data, "org").with_columns(
            pl.lit("original").alias("method")
        )
        kappa = pl.concat([base, test], how="vertical")
        dfs.append(kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    single_org = df.unique(["record", "dataset", "method", "id"], keep="any")
    result = single_org.select("dataset", "method", "id", "record", "kappa")
    return result


def _plot_kappa_vs_methods(data: pl.DataFrame, show: bool = False) -> None:
    dataset: str = data.item(0, "dataset")

    base_color = BASE_COLOR[dataset.lower()]
    highlight_color = HIGHLIGHT_COLOR[dataset.lower()]

    kappa_mean = data.group_by("method").agg(pl.mean("kappa")).sort("kappa")
    method_order = kappa_mean.get_column("method").to_list()

    palette = {
        method: (highlight_color if method == "original" else base_color)
        for method in method_order
    }

    plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07)
    sns.set_theme(style="whitegrid", context="paper")
    ax = sns.boxplot(
        data=data,
        x="method",
        y="kappa",
        showfliers=False,
        fill=True,
        order=method_order,
        palette=palette,
    )
    sns.swarmplot(
        data=data,
        x="method",
        y="kappa",
        color="black",
        alpha=0.7,
        size=3,
        order=method_order,
        ax=ax,
    )

    plt.suptitle(f"{dataset.upper()} - Kappa by Method", size=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Kappa", fontsize=12)
    plt.xticks(rotation=20, ha="center")
    plt.subplots_adjust(top=0.86, bottom=0.15)

    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(range(len(method_order)))
    secax.set_xticklabels(
        [f"{mean:.2f}" for mean in kappa_mean.get_column("kappa")],
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

    folder = os.path.join("figures", dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"figures/{dataset}_kappa_vs_method.png", dpi=300)

    if show:
        plt.show()
    pass


if __name__ == "__main__":
    plot_kappa_vs_methods(show=True)
