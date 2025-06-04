import re

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.plot.methods import METHODS, sort_dataframe_by_method_order
from src.utils.figures import adjust_axis_font, save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import extract_performance, extract_settings


def new_plot_delta_kappa_vs_train_size():
    df = _get_delta_kappa_data()
    delta_kappa_train_size_plot(df)


def _get_delta_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        test = extract_performance(method_data, "new").select(
            "dataset", "method", "record", "kappa", "key", "fold"
        )
        base = extract_performance(method_data, "org").select(
            "dataset",
            "method",
            "record",
            "key",
            "fold",
            pl.col("kappa").alias("base_kappa"),
        )
        delta_kappa = test.join(
            base, on=["dataset", "method", "record", "key", "fold"], how="left"
        ).with_columns((pl.col("kappa") - pl.col("base_kappa")).alias("delta_kappa"))

        train_size = extract_settings(method_data).select(
            "dataset", "key", "method", "train_size", "fold"
        )

        train_size_delta_kappa = delta_kappa.join(
            train_size, on=["dataset", "method", "key", "fold"], how="left"
        ).drop("key", "fold", "kappa", "base_kappa")

        dfs.append(train_size_delta_kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")

    valid_sizes = [4, 8, 16, 32, 64]

    def map_to_nearest_valid(ts):
        for v in valid_sizes:
            if abs(ts - v) <= 1:
                return v
        return None

    df_filtered = df.with_columns(
        [
            pl.col("train_size")
            .map_elements(map_to_nearest_valid, return_dtype=pl.Int64)
            .alias("train_size")
        ]
    ).filter(pl.col("train_size").is_not_null())
    # df = sort_dataframe_by_method_order(df)
    return sort_dataframe_by_method_order(df_filtered)


def extract_base_and_variant(method):
    match = re.match(r"([A-Za-z]+)(\d*)", method)
    if match:
        base = match.group(1)
        variant = match.group(2) if match.group(2) else None
        return base, variant
    return method, method


def delta_kappa_train_size_plot(df: pl.DataFrame):
    avg_df = (
        df.group_by(["method", "train_size"])
        .agg(pl.col("delta_kappa").mean().alias("delta_kappa_avg"))
        .with_columns(pl.col("delta_kappa_avg") * 100)
        .sort("train_size", "delta_kappa_avg")
        .with_columns(
            pl.col("method")
            .map_elements(lambda x: extract_base_and_variant(x)[0])
            .alias("base"),
            pl.col("method")
            .map_elements(lambda x: extract_base_and_variant(x)[1])
            .alias("variant"),
        )
        # .to_pandas()
    )

    train_size_order = {
        v: i for i, v in enumerate(sorted(avg_df["train_size"].unique()))
    }
    dfs = []
    for _, train_size_df in avg_df.group_by("train_size", maintain_order=True):
        dfs.append(
            train_size_df.with_row_index("xpos").with_columns(
                pl.col("train_size")
                .map_elements(lambda x: train_size_order[x], return_dtype=pl.Int64)
                .alias("ypos")
            )
        )

    data: pl.DataFrame = pl.concat(dfs, how="vertical")

    train_sizes = data.get_column("train_size").unique().sort()
    linestyles = {None: "solid", "10": "solid", "20": "dashed", "50": "dotted"}
    base_colors = dict(
        zip(
            data.get_column("base").unique().sort(),
            sns.color_palette("tab10"),
        )
    )
    square_width = 0.8
    square_height = 0.4
    fig = plt.figure(figsize=(15, 7))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.14)
    ax = plt.gca()

    for method, coords in data.group_by("method", maintain_order=True):
        xs_raw = coords.get_column("xpos")
        ys_raw = coords.get_column("ypos")

        ys, xs = [], []
        for x, y in zip(xs_raw, ys_raw):
            if y == min(ys_raw):
                xs.append(x)
                ys.append(y + square_height / 2)
            elif y == max(ys_raw):
                xs.append(x)
                ys.append(y - square_height / 2)
            else:
                xs.append(x)
                xs.append(x)
                ys.append(y - square_height / 2)
                ys.append(y + square_height / 2)

        variant = coords.item(0, "variant")
        base = coords.item(0, "base")
        ax.plot(
            xs,
            ys,
            linestyle=linestyles[variant],
            color=base_colors[base],
            label=method,
            zorder=1,
        )

    for row in data.iter_rows(named=True):
        print(row)
        # break
        x = row["xpos"]
        y = row["ypos"]
        val = row["delta_kappa_avg"]

        rect = plt.Rectangle(
            (x - square_width / 2, y - square_height / 2),
            square_width,
            square_height,
            color=base_colors[row["base"]],
            # alpha=0.7,
            zorder=2,
        )
        ax.add_patch(rect)

        plt.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=12, zorder=3)

    plt.yticks(range(len(train_size_order)), list(train_size_order.keys()))
    plt.xticks([])
    plt.ylabel("Train Size", fontsize=14)
    plt.title("Delta Kappa per Method based on Train Size", size=15)
    plt.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    first_half = METHODS[: len(METHODS) // 2]
    second_half = METHODS[len(METHODS) // 2 :]

    method_order = []
    for f, s in zip(first_half, second_half):
        method_order.append(f)
        method_order.append(s)

    method_order_iondex = {name: i for i, name in enumerate(method_order)}
    labels, handles = zip(
        *sorted(
            zip(labels, handles),
            key=lambda x: method_order_iondex.get(x[0], float("inf")),
        )
    )

    plt.legend(
        handles,
        labels,
        title="Method",
        title_fontsize=13,
        bbox_to_anchor=(0.5, -0.00),
        loc="upper center",
        ncol=11,
        frameon=False,
        prop={"size": 11},
    )
    adjust_axis_font(ax.yaxis, size=12)
    plt.tight_layout()

    save_figure(fig=fig, path="figures/delta_kappa_vs_train_size.png")


if __name__ == "__main__":
    new_plot_delta_kappa_vs_train_size()
