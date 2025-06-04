import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.plot.colors import BASE_COLOR
from src.utils.figures import save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance


def plot_pretrained_kappa_performance(show: bool = False) -> None:
    data = _get_pretrained_kappa_data()
    data = data.with_columns(pl.col("kappa") * 100)
    _plot_pretrained_kappa_performance(data, show=show)


def _get_pretrained_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue
        base = extract_performance(method_data, "org")
        dfs.append(base)

    concat: pl.DataFrame = pl.concat(dfs, how="vertical")
    single_org = concat.unique(["record", "dataset"], keep="any")
    result = single_org.select("dataset", "record", "kappa")
    # result = result.with_columns(pl.col("dataset").str.to_uppercase())
    return result.sort("dataset")


def _plot_pretrained_kappa_performance(data: pl.DataFrame, show: bool = False) -> None:
    kappa_info = data.group_by("dataset").agg(
        pl.col("kappa").mean().alias("mean_kappa"),
        pl.col("kappa").std().alias("std_kappa"),
        pl.col("kappa").max().alias("max_kappa"),
    )

    dataset_order = data.get_column("dataset").unique().sort().to_list()

    fig = plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=data.to_pandas(),
        x="dataset",
        y="kappa",
        hue="dataset",
        palette=BASE_COLOR,
        showfliers=False,
        order=dataset_order,
    )

    sns.swarmplot(
        data=data.to_pandas(),
        x="dataset",
        y="kappa",
        color="black",
        size=4,
        alpha=0.6,
        order=dataset_order,
    )

    bbox_right = 0.97
    bbox_left = 0.1
    center = (bbox_left + bbox_right) / 2
    plt.subplots_adjust(left=bbox_left, right=bbox_right, top=0.86, bottom=0.1)
    plt.suptitle("Kappa of Pretrained Model", size=14, x=center, ha="center")
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Kappa", fontsize=12)
    plt.ylim(top=100)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    secax = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))

    secax.set_xticks(range(len(dataset_order)))
    secax.set_xticklabels(
        [f"{mean:.2f}" for mean in kappa_info.get_column("mean_kappa")],
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    secax.set_xlabel("Mean Kappa", fontsize=12)

    save_figure(fig, "figures/original_kappa.png")

    if show:
        plt.show()

    pass


if __name__ == "__main__":
    plot_pretrained_kappa_performance()
    pass
