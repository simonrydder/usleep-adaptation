import os

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import MethodData, extract_performance


def prepare_data(data: list[MethodData]) -> pl.DataFrame:
    return (
        extract_performance(data, "org")
        .unique(["record", "dataset"], keep="any")
        .sort("dataset")
    )


def pretrained_kappa_performance_plot(
    data: list[MethodData], show: bool = False
) -> None:
    kappa = prepare_data(data)
    kappa_info = kappa.group_by("dataset").agg(
        pl.col("kappa").mean().alias("mean_kappa"),
        pl.col("kappa").std().alias("std_kappa"),
        pl.col("kappa").max().alias("max_kappa"),
    )

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=kappa,
        x="dataset",
        y="kappa",
        hue="dataset",
        palette="pastel",
        showfliers=False,
    )

    sns.swarmplot(
        data=kappa,
        x="dataset",
        y="kappa",
        color="black",
        size=4,
        alpha=0.6,
    )

    for i, row in enumerate(kappa_info.iter_rows(named=True)):
        box_color = ax.patches[i].get_facecolor()
        plt.text(
            i,
            0.9,
            f"{row['mean_kappa']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=box_color,
            fontdict={"weight": "bold"},
        )

    # Formatting
    plt.xlabel("Dataset")
    plt.ylabel("Kappa")
    plt.title("Kappa of Pretrained Model")
    plt.ylim(top=1)

    folder = os.path.join("figures")
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig("figures/original_kappa.png", dpi=300)

    if show:
        plt.show()
    pass


if __name__ == "__main__":
    data = load_data(ids=[3])
    pretrained_kappa_performance_plot(data, show=True)
    pass
