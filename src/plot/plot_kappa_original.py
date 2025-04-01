import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plot.utils.neptune_api import get_tag_data


def plot_delta_kappas_vs_methods(tag_ids: list[str]) -> None:
    """
    Plots Delta Kappa values for different methods using Seaborn boxplot and stripplot,
    aggregating data across all available folds (provided as a dictionary) for each tag ID.

    Args:
        tag_ids: A list of tag IDs to fetch data for.
    """
    plot_data: list = []  # List to store data for DataFrame creation

    for tag_id in tag_ids:
        tag_data_dict = get_tag_data(tag_id)

        if not tag_data_dict:
            print(f"Warning: No data found for tag_id {tag_id}")
            continue
        for fold_index, tag_data in tag_data_dict.items():
            try:
                method = tag_data.config.experiment.method
                dataset_name = tag_data.config.experiment.dataset

                kappas = [
                    kappa_old.value for kappa_old in tag_data.new_performance.kappa
                ]
                for kap in kappas:
                    plot_data.append(
                        {
                            "Method": method,
                            "Kappa": kap,
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
    df = pd.DataFrame(plot_data)
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 7))
    sns.boxplot(x="Method", y="Kappa", data=df, palette="pastel", showfliers=False)

    sns.stripplot(
        x="Method",
        y="Kappa",
        data=df,
        color="black",
        jitter=True,
        size=4,
        alpha=0.6,
    )

    plt.xlabel("Adapter Method")
    plt.ylabel("Kappa")
    plt.title(
        f"Kappa Score by Method \nDataset: {dataset_name}"  # type: ignore
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_delta_kappas_vs_methods(["1aYTnFXcM", "1wHtRWs0", "fctKJw55"])
