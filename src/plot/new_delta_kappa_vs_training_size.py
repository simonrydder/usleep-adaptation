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
        .to_pandas()
    )
    train_sizes = sorted(avg_df["train_size"].unique())

    dynamic_positions = {}
    for t in train_sizes:
        subset = avg_df[avg_df["train_size"] == t].copy()
        subset = subset.sort_values("delta_kappa_avg")  # Low to high
        for xpos, row in enumerate(subset.itertuples()):
            dynamic_positions[(row.method, row.train_size)] = xpos

    methods = df["method"].unique().to_list()

    method_bases = {m: extract_base_and_variant(m)[0] for m in methods}
    method_variants = {m: extract_base_and_variant(m)[1] for m in methods}

    base_methods = sorted(set(method_bases.values()))
    base_colors = dict(zip(base_methods, sns.color_palette("tab10")))

    linestyles = {None: "solid", "10": "solid", "20": "dashed", "50": "dotted"}
    style_map = {m: linestyles.get(v) for m, v in method_variants.items()}

    fig = plt.figure(figsize=(15, 7))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.14)
    ax = plt.gca()

    method_tracks = {method: [] for method in methods}

    for row in avg_df.itertuples():
        x = dynamic_positions[(row.method, row.train_size)]
        y = train_sizes.index(row.train_size)
        method_tracks[row.method].append((x, y))

        rect = plt.Rectangle(
            (x - 0.4, y - 0.2),
            0.8,
            0.4,
            color=base_colors[method_bases[row.method]],
            alpha=0.7,
        )
        ax.add_patch(rect)

        plt.text(
            x, y, f"{row.delta_kappa_avg:.2f}", ha="center", va="center", fontsize=12
        )

    for method, coords in method_tracks.items():
        coords_sorted = sorted(coords, key=lambda x: x[1])
        xs, ys = zip(*coords_sorted)
        plt.plot(
            xs,
            ys,
            linestyle=style_map[method],
            color=base_colors[method_bases[method]],
            label=method,
        )

    plt.yticks(range(len(train_sizes)), train_sizes)
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
