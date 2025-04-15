import os

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.neptune_api.method_data import MethodData, extract_performance


def plot_validation_delta_kappa(
    data: dict[str, MethodData], dataset: str, show: bool = False
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

    sns.set_theme(style="whitegrid")

    g = sns.FacetGrid(
        epoch_df,
        col="method",
        col_wrap=3,
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
        color="gray",
        linewidth=1,
        alpha=0.4,
    )
    for ax, method in zip(g.axes.flatten(), g.col_names):
        method_df = fold_avg[fold_avg["method"] == method]
        sns.lineplot(
            data=method_df,
            x="epoch",
            y="kappa",
            ax=ax,
            color="C0",  # Or pick your own color
            linewidth=2.5,
            label="Mean",
        )
        ax.legend(loc="lower right")

    g.set_axis_labels("Epoch", "Kappa")
    g.figure.subplots_adjust(top=0.95)
    g.figure.suptitle(f"{dataset} Validation Kappa vs Epoch for methods")
    # plt.tight_layout()

    folder = os.path.join("figures", dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"figures/{dataset}/validation_kappa.png", dpi=300)

    if show:
        plt.show()


def prepare_data(data: dict[str, MethodData]) -> pl.DataFrame:
    val = extract_validation_data(data)
    base = extract_performance(data, mode="org")

    delta_val = (
        val.join(base, on=["method", "record"], how="left", suffix="_base")
        .with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))
        .select("step", "epoch", "method", "fold", "record", "delta_kappa", "kappa")
    )
    return delta_val


def extract_validation_data(data: dict[str, MethodData]) -> pl.DataFrame:
    dfs = []
    for method, method_data in data.items():
        for fold, fold_data in method_data.folds.items():
            validation = pl.DataFrame(fold_data.validation_step)
            validation = (
                validation.with_columns(
                    pl.lit(method).alias("method"),
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

    data = load_data("eesm19")
    plot_validation_delta_kappa(data, "eesm19", show=True)
