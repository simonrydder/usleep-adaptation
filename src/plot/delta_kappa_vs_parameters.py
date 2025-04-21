import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.utils.neptune_api.method_data import (
    MethodData,
    extract_performance,
    extract_settings,
)


def prepare_data(data: list[MethodData]) -> pl.DataFrame:
    base = extract_performance(data, "org")
    test = extract_performance(data, "new")

    delta_kappa = (
        (
            test.join(base, on=["method", "record"], how="left", suffix="_base")
            .with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))
            .select("method", "record", "delta_kappa", "kappa", "kappa_base")
        )
        .group_by("method")
        .agg(pl.col("delta_kappa").mean())
    )

    settings = extract_settings(data)
    assert settings.n_unique("method") == settings.n_unique(
        ["method", "free_parameters", "total_parameters"]
    )

    settings = settings.select("method", "free_parameters").unique(keep="any")

    result = delta_kappa.join(settings, on="method").with_columns(
        pl.col("method").str.strip_chars(" 0123456789").alias("method_type")
    )
    return result.sort("method")


def plot_delta_kappa_vs_parameters(
    data: list[MethodData], dataset: str, show: bool = False
) -> None:
    df = prepare_data(data)
    df_pandas = df.to_pandas()

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", context="paper")

    sns.scatterplot(
        data=df_pandas,
        x="free_parameters",
        y="delta_kappa",
        hue="method_type",
        # palette=sns.color_palette("tab20"),
    )

    plt.xscale("log")
    plt.xlim(left=10)

    plt.title(f"{dataset} - Delta Kappa vs. Number of Free Parameters")
    plt.xlabel("Number of Free Parameters")
    plt.ylabel("Delta Kappa")
    plt.grid(True)
    plt.tight_layout()

    folder = os.path.join("figures", dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"figures/{dataset}/delta_kappa_vs_parameters.png", dpi=300)

    if show:
        plt.show()

    return None


if __name__ == "__main__":
    from src.utils.neptune_api.data_loader import load_data

    data = load_data("eesm19")
    plot_delta_kappa_vs_parameters(data, "eesm19", show=True)
