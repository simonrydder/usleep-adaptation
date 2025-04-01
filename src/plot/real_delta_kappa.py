import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
