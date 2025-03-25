import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    pass


if __name__ == "__main__":
    # Example dictionary structure
    data = {
        "bitfit": {
            "eesm19": [1.2, 2.3, 1.8, 2.1],
            "svuh": [3.1, 3.5, 2.9, 3.3, 3.8],
        },
        "fish": {
            "eesm19": [1.5, 2.1, 1.7, 2.0],
            "svuh": [3.0, 3.4, 3.1, 3.2, 3.6],
        },
    }

    # Convert dictionary to a long-form DataFrame
    df_list = []
    for method, datasets in data.items():
        for dataset, values in datasets.items():
            for value in values:
                df_list.append({"Method": method, "Dataset": dataset, "Value": value})

    df = pd.DataFrame(df_list)

    # Plot for each dataset
    for dataset in df["Dataset"].unique():
        plt.figure(figsize=(8, 6))

        subset = df[df["Dataset"] == dataset]

        sns.boxplot(
            x="Method",
            y="Value",
            data=subset,
            width=0.5,
            showcaps=False,
            boxprops={"facecolor": "None"},
        )
        sns.stripplot(
            x="Method", y="Value", data=subset, jitter=True, size=8, alpha=0.6
        )

        plt.title(f"Boxplot with Points for {dataset}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        x="Method",
        y="Value",
        hue="Dataset",
        data=df,
        dodge=True,
        showcaps=False,
        boxprops={"facecolor": "None"},
        width=0.6,
    )

    sns.stripplot(
        x="Method",
        y="Value",
        hue="Dataset",
        data=df,
        dodge=True,
        jitter=True,
        size=8,
        alpha=0.7,
        marker="o",
        linewidth=1,
        edgecolor="black",
        legend=False,
    )

    plt.title("Boxplot with Points for Each Method and Dataset")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Dataset")
    plt.show()

    pass
