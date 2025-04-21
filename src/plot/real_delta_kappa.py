import os

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from src.utils.neptune_api.method_data import MethodData, extract_performance


def plot_performance_delta_kappa_vs_methods(
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


def plot_delta_kappas_vs_methods(data: dict) -> None:
    """
    Plots Delta Kappa values for different methods using Seaborn boxplot and stripplot,
    aggregating data across all available folds (provided as a dictionary) for each tag ID.

    Args:
        tag_ids: A list of tag IDs to fetch data for.
    """
    plot_data: list = []  # List to store data for DataFrame creation
    for id in data.keys():
        for fold_index, tag_data in data[id].items():
            try:
                method = tag_data.config.experiment.method
                dataset_name = tag_data.config.experiment.dataset

                delta_kappas = [
                    kappa_new.value - kappa_old.value
                    for kappa_new, kappa_old in zip(
                        tag_data.new_performance.kappa,
                        tag_data.original_performance.kappa,
                    )
                ]
                for dk in delta_kappas:
                    plot_data.append(
                        {
                            "Method": method,
                            "Delta Kappa": dk,
                            "TagID": id,
                            "Fold": fold_index,
                        }
                    )

            except Exception as e:
                print(
                    f"Warning: An unexpected error occurred processing fold {fold_index} for tag {id}: {e}"
                )
    if not plot_data:
        print("No data available to plot.")
        return
    df = pd.DataFrame(plot_data)
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 7))
    sns.boxplot(
        x="Method", y="Delta Kappa", data=df, palette="pastel", showfliers=False
    )

    sns.stripplot(
        x="Method",
        y="Delta Kappa",
        data=df,
        color="black",
        jitter=True,
        size=4,
        alpha=0.6,
    )

    plt.xlabel("Adapter Method")
    plt.ylabel("Delta Kappa (New Kappa - Old Kappa)")
    plt.title(
        f"Change in Kappa Score by Method \nDataset: {dataset_name}"  # type: ignore
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("fig/delta_kap_vs_methods_boxplot.png")


if __name__ == "__main__":
    from src.utils.neptune_api.data_loader import load_data

    data = load_data("eesm19")
    plot_performance_delta_kappa_vs_methods(data, "eesm19", show=True)
