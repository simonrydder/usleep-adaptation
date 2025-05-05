import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src.plot.colors import HIGHLIGHT_COLOR
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_validation_data


def plot_seeding_distribution() -> None:
    data = _get_seeding_data()
    _plot_seeding_distribution(data, 2)


def _get_seeding_data() -> pl.DataFrame:
    raw_data = load_data(datasets=["eesm19"], methods=["Full"])

    dfs = []
    for data in raw_data:
        val = extract_validation_data(data)
        val = val.drop("accuracy", "loss", "f1")
        dfs.append(val)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")

    return df.filter(pl.col("key").is_in(["RAbevDTr", "Eo974696"]))


def _plot_seeding_distribution(data: pl.DataFrame, cols: int) -> None:
    dataset: str = data.item(0, "dataset")
    color = HIGHLIGHT_COLOR[dataset.lower()]

    epoch_df = (
        data.group_by("epoch", "fold", "key").mean().drop("step", "record").sort("key")
    )
    fold_avg = epoch_df.group_by("epoch", "key").agg(pl.mean("kappa"))

    g = sns.FacetGrid(
        epoch_df,
        col="key",
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

    for ax, key in zip(g.axes.flatten(), g.col_names):
        assert isinstance(ax, Axes)
        key_df = fold_avg.filter(pl.col("key") == key)
        sns.lineplot(
            data=key_df,
            x="epoch",
            y="kappa",
            ax=ax,
            color=color,
            linewidth=2,
        )

        ax.set_ylim((0.0, 0.9))

    g.set_axis_labels("Epoch", "Kappa", fontsize=10)
    g.figure.subplots_adjust(top=0.81)
    g.figure.suptitle(
        f"{dataset.upper()} - Validation Kappa for each Method", fontsize=14
    )

    plt.show()


if __name__ == "__main__":
    plot_seeding_distribution()
