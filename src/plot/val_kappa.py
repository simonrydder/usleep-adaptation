import os

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src.utils.neptune_api.method_data import MethodData, extract_performance


def plot_validation_kappa(
    data: list[MethodData], dataset: str, show: bool = False, cols: int = 5
) -> None:
    df = prepare_data(data)
    epoch_df = (
        df.group_by("epoch", "method", "fold")
        .mean()
        .drop("step", "record")
        .sort("method")
    )
    fold_avg = epoch_df.group_by("epoch", "method").agg(pl.mean("kappa"))

    epoch_df = epoch_df.to_pandas()
    fold_avg = fold_avg.to_pandas()

    sns.set_theme(style="whitegrid", context="paper")

    g = sns.FacetGrid(
        epoch_df,
        col="method",
        col_wrap=cols,
        sharey=True,
        sharex=True,
        height=3.5,
        aspect=1.5,
    )
    g.set_titles(template="{col_name}")

    g.map_dataframe(
        sns.lineplot,
        x="epoch",
        y="kappa",
        units="fold",
        estimator=None,
        color="black",
        linewidth=1,
        alpha=0.5,
    )
    for ax, method in zip(g.axes.flatten(), g.col_names):
        assert isinstance(ax, Axes)
        method_df = fold_avg[fold_avg["method"] == method]
        sns.lineplot(
            data=method_df,
            x="epoch",
            y="kappa",
            ax=ax,
            color="C0",  # Or pick your own color
            linewidth=2,
            label="Mean",
        )
        ax.legend(loc="lower right")
        ax.set_ylim((0.0, 0.85))

    g.set_axis_labels("Epoch", "Kappa")
    g.figure.subplots_adjust(top=0.95)
    g.figure.suptitle(f"{dataset} - Validation Kappa vs Epoch for methods")
    # plt.tight_layout()

    folder = os.path.join("figures", dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"figures/{dataset}/validation_kappa.png", dpi=300)

    if show:
        plt.show()


def prepare_data(data: list[MethodData]) -> pl.DataFrame:
    val = extract_validation_data(data)
    base = extract_performance(data, mode="org")

    delta_val = (
        val.join(base, on=["method", "record"], how="left", suffix="_base")
        .with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))
        .select("step", "epoch", "method", "fold", "record", "delta_kappa", "kappa")
    )
    return delta_val


def extract_validation_data(data: list[MethodData]) -> pl.DataFrame:
    dfs = []
    for method_data in data:
        for fold, fold_data in method_data.folds.items():
            validation = pl.DataFrame(fold_data.validation_step)
            validation = (
                validation.with_columns(
                    pl.lit(method_data.method).alias("method"),
                    pl.lit(fold).alias("fold"),
                )
                .with_row_index("step")
                .with_columns(
                    (pl.col("step") // fold_data.config.data.sizes.validation).alias(
                        "epoch"
                    )
                )
            )
            dfs.append(validation)

    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":
    from src.utils.neptune_api.data_loader import load_data

    data = load_data(
        datasets=["eesm19"], ids=[3], methods=["BitFit", "LoRA20", "PCL20"]
    )
    plot_validation_kappa(data, "eesm19", show=True, cols=1)
