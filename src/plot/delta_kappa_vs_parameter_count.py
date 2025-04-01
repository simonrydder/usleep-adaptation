import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_delta_kappas_vs_parameter_percentage(data: dict) -> None:
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
