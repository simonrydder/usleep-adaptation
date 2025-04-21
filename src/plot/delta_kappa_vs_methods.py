import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.utils.neptune_api.method_data import MethodData, extract_performance


def plot_delta_kappa_vs_methods(
    data: dict[str, MethodData], dataset: str, show: bool = False
) -> None:
    df = prepare_data(data)
    df = df.to_pandas()

    plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07)
    sns.set_theme(style="whitegrid", context="paper")
    sns.boxplot(data=df, x="method", y="delta_kappa", showfliers=False, fill=True)
    sns.swarmplot(
        data=df, x="method", y="delta_kappa", color="black", alpha=0.7, size=3
    )

    # group_means = df.groupby("method")["delta_kappa"].mean()

    plt.title(f"{dataset} - Delta Kappa by Method", fontdict={"size": 14})
    plt.xlabel("Method")
    plt.ylabel("Delta Kappa")
    # plt.tight_layout()

    folder = os.path.join("figures", dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(f"figures/{dataset}/delta_kappa_vs_method.png", dpi=300)

    if show:
        plt.show()
    pass


def prepare_data(data: dict[str, MethodData]) -> pl.DataFrame:
    base = extract_performance(data, "org")
    test = extract_performance(data, "new")

    delta_kappa = (
        test.join(base, on=["method", "record"], how="left", suffix="_base")
        .with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))
        .select("method", "record", "delta_kappa", "kappa", "kappa_base")
    )

    return delta_kappa


if __name__ == "__main__":
    from src.utils.neptune_api.data_loader import load_data

    data = load_data("eesm19")
    plot_delta_kappa_vs_methods(data, "eesm19", show=True)
