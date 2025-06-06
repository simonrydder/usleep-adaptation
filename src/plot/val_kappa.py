import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src.plot.colors import HIGHLIGHT_COLOR
from src.plot.methods import sort_dataframe_by_method_order
from src.utils.figures import adjust_axis_font, save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import (
    extract_validation_data,
)


def plot_validation_kappa(cols: int = 4, show: bool = False) -> None:
    df = _get_validation_kappa_data()

    for _, grp_df in df.group_by("dataset", maintain_order=True):
        _plot_validation_kappa(grp_df, cols=cols, show=show)


def _get_validation_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue

        val = extract_validation_data(method_data)
        val = val.drop("accuracy", "loss", "f1")
        dfs.append(val)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    df = df.with_columns(pl.col("kappa") * 100)
    return sort_dataframe_by_method_order(df)


def _plot_validation_kappa(data: pl.DataFrame, cols: int, show: bool) -> None:
    dataset: str = data.item(0, "dataset")
    color = HIGHLIGHT_COLOR[dataset.lower()]

    epoch_df = (
        data.group_by("epoch", "method", "fold", maintain_order=True)
        .mean()
        .drop("step", "record")
        # .sort("method")
    )
    fold_avg = epoch_df.group_by("epoch", "method", maintain_order=True).agg(
        pl.mean("kappa")
    )
    max_epoch_mean = (
        epoch_df.group_by("method", "fold", maintain_order=True)
        .agg(pl.col("epoch").max())
        .group_by("method", maintain_order=True)
        .agg(pl.col("epoch").mean().alias("max_epoch_mean"))
    )

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.FacetGrid(
        epoch_df,
        col="method",
        col_wrap=cols,
        sharey=True,
        sharex=True,
        height=3,
        aspect=1.25,
    )
    g.set_titles(template="{col_name}", size=15)

    g.map_dataframe(
        sns.lineplot,
        x="epoch",
        y="kappa",
        units="fold",
        estimator=None,
        color="gray",
        linewidth=1,
        alpha=0.5,
    )
    for i, (ax, method) in enumerate(zip(g.axes.flatten(), g.col_names)):
        assert isinstance(ax, Axes)
        epoch_limit = max_epoch_mean.filter(pl.col("method") == method).item(
            0, "max_epoch_mean"
        )
        method_df = fold_avg.filter(
            pl.col("method") == method, pl.col("epoch") <= int(round(epoch_limit, 0))
        )
        sns.lineplot(
            data=method_df,
            x="epoch",
            y="kappa",
            ax=ax,
            color=color,
            linewidth=2,
            label="Mean" if i == 0 else None,
        )
        if i == 0:
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.99, 0.98),
                bbox_transform=g.figure.transFigure,
                prop={"size": 14},
            )

        ax.set_ylim((-5, 95))

        adjust_axis_font(ax.yaxis, size=12)
        adjust_axis_font(ax.xaxis, size=12)

    g.set_axis_labels("Epoch", "Kappa", fontsize=14)
    g.figure.subplots_adjust(top=0.925)
    g.figure.suptitle(
        f"{dataset.upper()} - Validation Kappa for each Method", fontsize=18
    )

    save_figure(g.figure, f"figures/{dataset}_validation_kappa.png")

    if show:
        plt.show()


if __name__ == "__main__":
    plot_validation_kappa()
