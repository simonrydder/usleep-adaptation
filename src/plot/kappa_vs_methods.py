import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.plot.colors import BASE_COLOR, HIGHLIGHT_COLOR
from src.utils.figures import save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance


def plot_kappa_vs_methods(show: bool = False) -> None:
    data = get_kappa_data()
    data = data.with_columns(pl.col("kappa") * 100)

    for _, grp_df in data.group_by("dataset"):
        _plot_kappa_vs_methods(grp_df, show=show)


def get_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue

        test = extract_performance(method_data, "new")
        base = extract_performance(method_data, "org").with_columns(
            pl.lit("original").alias("method")
        )
        kappa = pl.concat([base, test], how="vertical")
        dfs.append(kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    single_org = df.unique(["record", "dataset", "method"], keep="any")
    result = single_org.select("dataset", "method", "key", "record", "kappa")
    return result


def _plot_kappa_vs_methods(data: pl.DataFrame, show: bool = False) -> None:
    dataset: str = data.item(0, "dataset")

    base_color = BASE_COLOR[dataset.lower()]
    highlight_color = HIGHLIGHT_COLOR[dataset.lower()]
    highlight_color = "white"

    kappa_mean = data.group_by("method").agg(pl.mean("kappa")).sort("kappa")
    method_order = kappa_mean.get_column("method").to_list()

    palette = {
        method: (highlight_color if method == "original" else base_color)
        for method in method_order
    }

    fig = plt.figure(figsize=(18, 7))
    sns.set_theme(style="whitegrid", context="paper")
    ax = sns.boxplot(
        data=data,
        x="method",
        y="kappa",
        showfliers=False,
        fill=True,
        order=method_order,
        hue="method",
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

    bbox_right = 0.97
    bbox_left = 0.05
    center = (bbox_left + bbox_right) / 2
    plt.subplots_adjust(left=bbox_left, right=bbox_right, top=0.85, bottom=0.12)
    plt.suptitle(
        f"{dataset.upper()} - Kappa vs. Method", size=16, x=center, ha="center"
    )
    plt.xlabel("Method", fontsize=14)
    plt.ylabel("Kappa", fontsize=14)
    plt.xticks(rotation=20, ha="center")

    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    for label in ax.get_xticklabels():
        if label.get_text() == "original":
            label.set_fontweight("bold")

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(range(len(method_order)))
    secax.set_xticklabels(
        [f"{mean:.2f}" for mean in kappa_mean.get_column("kappa")],
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    secax.set_xlabel("Mean Kappa", fontsize=14)

    save_figure(fig=fig, path=f"figures/{dataset}_kappa_vs_method.png")

    if show:
        plt.show()
    pass


if __name__ == "__main__":
    plot_kappa_vs_methods()
