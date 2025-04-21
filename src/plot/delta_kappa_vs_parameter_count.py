import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from src.utils.neptune_api.method_data import (
    MethodData,
    extract_performance,
    extract_settings,
)


def prepare_data(data: dict[str, MethodData]) -> pl.DataFrame:
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
    return result


def plot_delta_kappas_vs_parameter(
    data: dict[str, MethodData], dataset: str, show: bool = False
) -> None:
    df = prepare_data(data)
    df_pandas = df.to_pandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_pandas,
        x="free_parameters",
        y="delta_kappa",
        hue="method_type",
        palette=sns.color_palette("tab20"),
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
    """
    Plots the improvement in kappa (delta kappa) versus the free parameter count
    for different finetuning methods.

    Args:
        tag_ids: A list of tag IDs to fetch data for.
    """
    plot_data = []  # List to store data for DataFrame creation
    for id in data.keys():
        for fold_index, tag_data in data[id].items():
            try:
                method = tag_data.config.experiment.method
                dataset_name = tag_data.config.experiment.dataset

                # Calculate improvement in kappa (delta kappa)
                delta_kappas = [
                    kappa_new.value - kappa_old.value
                    for kappa_new, kappa_old in zip(
                        tag_data.new_performance.kappa,
                        tag_data.original_performance.kappa,
                    )
                ]
                parameters_free_percentage = 100 * (
                    1
                    - (
                        (
                            np.abs(
                                tag_data.parameters.model.free
                                - tag_data.parameters.total
                            )
                        )
                        / (tag_data.parameters.total)
                    )
                )

                for dk in delta_kappas:
                    plot_data.append(
                        {
                            "Method": method,
                            "Delta Kappa": dk,
                            "Parameter Free Percentage": parameters_free_percentage,
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

    # Create DataFrame from collected data
    df = pd.DataFrame(plot_data)

    df = (
        df.groupby("Method")
        .agg({"Delta Kappa": "mean", "Parameter Free Percentage": "mean"})
        .reset_index()
    )
    # --- Create Scatter Plot with Optional Log Scale and Regression Line ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))

    # Create scatter plot
    ax = sns.scatterplot(
        data=df,
        x="Parameter Free Percentage",
        y="Delta Kappa",
        hue="Method",  # Color points by method
        style="Method",  # Different markers per method
        s=80,  # Adjust point size
        alpha=0.8,  # Set transparency
    )
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    # Option: To add trend lines, you can loop over methods and use regplot
    methods = df["Method"].unique()
    for m in methods:
        method_data = df[df["Method"] == m]
        sns.regplot(
            data=method_data,
            x="Parameter Free Percentage",
            y="Delta Kappa",
            scatter=False,
            ax=ax,
            label=f"{m} Trend",
        )

    plt.xlabel("Parameter Free Percentage")
    plt.ylabel("Delta Kappa (New Kappa - Old Kappa)")
    plt.title(
        f"Delta Kappa vs. Parameter Free Percentage by Method \nDataset: {dataset_name}"  # type: ignore
    )  # type: ignore
    # Adjust legend to combine scatter and trend lines neatly
    plt.legend(title="Adapter Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("fig/delta_kap_vs_param_perc.png")


if __name__ == "__main__":
    from src.utils.neptune_api.data_loader import load_data

    data = load_data("eesm19")
    plot_delta_kappas_vs_parameter(data, "eesm19", show=True)
