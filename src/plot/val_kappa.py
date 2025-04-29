import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src.plot.colors import HIGHLIGHT_COLOR
from src.utils.figures import save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import (
    extract_validation_data,
)


def plot_validation_kappa(cols: int = 4, show: bool = False) -> None:
    df = _get_validation_kappa_data()

    for _, grp_df in df.group_by("dataset"):
        _plot_validation_kappa(grp_df, cols=cols, show=show)


def _get_validation_kappa_data() -> pl.DataFrame:
    raw_data = load_data(ids=[0])

    dfs = []
    for method_data in raw_data:
        val = extract_validation_data(method_data)
        val = val.drop("accuracy", "loss", "f1", "id")
        dfs.append(val)

    df = pl.concat(dfs, how="vertical")

    return df


def _plot_validation_kappa(data: pl.DataFrame, cols: int, show: bool) -> None:
    dataset: str = data.item(0, "dataset")
    color = HIGHLIGHT_COLOR[dataset.lower()]

    epoch_df = (
        data.group_by("epoch", "method", "fold")
        .mean()
        .drop("step", "record")
        .sort("method")
    )
    fold_avg = epoch_df.group_by("epoch", "method").agg(pl.mean("kappa"))

    epoch_df = epoch_df
    fold_avg = fold_avg

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.FacetGrid(
        epoch_df,
        col="method",
        col_wrap=cols,
        sharey=True,
        sharex=True,
        height=2.5,
        aspect=1.5,
    )
    g.set_titles(template="{col_name}", size=11)

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
        method_df = fold_avg.filter(pl.col("method") == method)
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
                bbox_to_anchor=(0.99, 0.97),
                bbox_transform=g.figure.transFigure,
                prop={"size": 10},
            )

        ax.set_ylim((0.0, 0.9))

    g.set_axis_labels("Epoch", "Kappa", fontsize=10)
    g.figure.subplots_adjust(top=0.925)
    g.figure.suptitle(
        f"{dataset.upper()} - Validation Kappa for each Method", fontsize=14
    )

    save_figure(g.figure, f"figures/{dataset}_validation_kappa.png")

    if show:
        plt.show()


if __name__ == "__main__":
    plot_validation_kappa(show=True, cols=3)
