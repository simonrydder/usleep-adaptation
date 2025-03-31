import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plot.utils.neptune_api import get_tag_data


def plot_delta_kappas_vs_parameter_count(tag_ids: list[str]) -> None:
    """
    Plots the improvement in kappa (delta kappa) versus the free parameter count
    for different finetuning methods.

    Args:
        tag_ids: A list of tag IDs to fetch data for.
    """
    plot_data = []  # List to store data for DataFrame creation

    for tag_id in tag_ids:
        tag_data_dict = get_tag_data(tag_id)

        if not tag_data_dict:
            print(f"Warning: No data found for tag_id {tag_id}")
            continue

        for fold_index, tag_data in tag_data_dict.items():
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
                parameter_count = tag_data.parameters.model.free

                for dk in delta_kappas:
                    plot_data.append(
                        {
                            "Method": method,
                            "Delta Kappa": dk,
                            "Parameter Count": parameter_count,
                            "TagID": tag_id,
                            "Fold": fold_index,
                        }
                    )
            except Exception as e:
                print(
                    f"Warning: An unexpected error occurred processing fold {fold_index} for tag {tag_id}: {e}"
                )

    if not plot_data:
        print("No data available to plot.")
        return

    # Create DataFrame from collected data
    df = pd.DataFrame(plot_data)

    df = (
        df.groupby("Method")
        .agg({"Delta Kappa": "mean", "Parameter Count": "mean"})
        .reset_index()
    )
    # --- Create Scatter Plot with Optional Log Scale and Regression Line ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))

    # Create scatter plot
    ax = sns.scatterplot(
        data=df,
        x="Parameter Count",
        y="Delta Kappa",
        hue="Method",  # Color points by method
        style="Method",  # Different markers per method
        s=80,  # Adjust point size
        alpha=0.8,  # Set transparency
    )

    # Option: To add trend lines, you can loop over methods and use regplot
    methods = df["Method"].unique()
    for m in methods:
        method_data = df[df["Method"] == m]
        sns.regplot(
            data=method_data,
            x="Parameter Count",
            y="Delta Kappa",
            scatter=False,
            ax=ax,
            label=f"{m} Trend",
        )

    plt.xlabel("Parameter Count")
    plt.ylabel("Delta Kappa (New Kappa - Old Kappa)")
    plt.title(f"Delta Kappa vs. Parameter Count by Method \nDataset: {dataset_name}")  # type: ignore
    # Adjust legend to combine scatter and trend lines neatly
    plt.legend(title="Adapter Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_delta_kappas_vs_parameter_count(["1aYTnFXcM", "1wHtRWs0"])
    plot_delta_kappas_vs_parameter_count(["1aYTnFXcM", "1wHtRWs0"])
