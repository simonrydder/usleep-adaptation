import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.plot.colors import HIGHLIGHT_COLOR
from src.utils.figures import save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import (
    extract_performance,
    extract_settings,
)


def plot_delta_kappa_vs_train_size(show: bool = False) -> None:
    data = _get_delta_kappa_data()
    _delta_kappa_vs_train_size_method(data, col_wrap=5)
    _delta_kappa_vs_train_size_dataset(data, col_wrap=2)


def _get_delta_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        test = extract_performance(method_data, "new").drop("accuracy", "loss", "f1")
        base = extract_performance(method_data, "org").drop("accuracy", "loss", "f1")

        delta_kappa = (
            test.join(
                base,
                on=["method", "record", "dataset", "key"],
                how="inner",
                suffix="_base",
            )
            .with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))
            .select(
                "key", "dataset", "method", "record", "fold", "kappa", "delta_kappa"
            )
        )

        train_size = extract_settings(method_data).select(
            "dataset", "key", "method", "train_size", "fold"
        )

        train_size_delta_kappa = delta_kappa.join(
            train_size, on=["dataset", "method", "key", "fold"], how="left"
        )

        dfs.append(train_size_delta_kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    delta_kappa_mean = df.group_by(["dataset", "method", "key"]).agg(
        pl.col("kappa").mean(),
        pl.col("delta_kappa").mean(),
        pl.col("train_size").max(),
    )

    return delta_kappa_mean.sort("dataset", "method")


def _delta_kappa_vs_train_size_dataset(data: pl.DataFrame, col_wrap: int):
    g = sns.FacetGrid(
        data,
        col="dataset",
        sharex=True,
        sharey=True,
        height=4,
        aspect=1.5,
        col_wrap=col_wrap,
    )

    def plot_mapping(data: pl.DataFrame | pd.DataFrame, **kwargs):
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        ax = plt.gca()

        sns.lineplot(
            data=data,
            x="train_size",
            y="delta_kappa",
            estimator="mean",
            errorbar=("se", 2),
            err_style="bars",
            hue="dataset",
            palette=HIGHLIGHT_COLOR,
            alpha=0.7,
            marker=".",
            markersize=10,
            ax=ax,
        )

        ax.grid(True)
        # ax.spines["right"].set_visible(True)
        # ax.spines["top"].set_visible(True)

    g.map_dataframe(plot_mapping)

    g.set_titles(template="{col_name}", size=13)

    g.figure.suptitle(
        "Delta Kappa vs. Number of Nights in Train by Dataset", fontsize=15
    )

    save_figure(g.figure, "figures/delta_kappa_vs_train_size_dataset.png")


def _delta_kappa_vs_train_size_method(data: pl.DataFrame, col_wrap: int):
    g = sns.FacetGrid(
        data,
        col="method",
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.5,
        col_wrap=col_wrap,
    )

    delta_kappa = data.get_column("delta_kappa")
    _min: float = delta_kappa.min()  # type: ignore
    _max: float = delta_kappa.max()  # type: ignore

    def plot_mapping(data: pl.DataFrame | pd.DataFrame, **kwargs):
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

        ax = plt.gca()

        # sns.regplot(
        #     data=data,
        #     x="train_size",
        #     y="delta_kappa",
        #     marker="",
        #     ax=ax,
        # )
        sns.lineplot(
            data=data,
            x="train_size",
            y="delta_kappa",
            estimator="mean",
            errorbar=("se", 2),
            err_style="bars",
            hue="dataset",
            palette=HIGHLIGHT_COLOR,
            alpha=0.6,
            marker=".",
            markersize=10,
            ax=ax,
        )
        ax.set_ylim(kwargs["ylim"])

        ax.grid(True)
        # ax.spines["right"].set_visible(True)
        # ax.spines["top"].set_visible(True)

    g.map_dataframe(plot_mapping, ylim=(-1.05 * _max, 1.05 * _max))

    g.set_titles(template="{col_name}", size=13)

    g.figure.suptitle(
        "Delta Kappa vs. Number of Nights in Train by Method", fontsize=15
    )
    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07)

    save_figure(g.figure, "figures/delta_kappa_vs_train_size_method.png")


if __name__ == "__main__":
    plot_delta_kappa_vs_train_size(show=True)
