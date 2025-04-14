import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt

from src.plot.utils.neptune_api_old import RunData, get_original, get_tag_data


def pretrained_kappa_performance_plot(data: dict[str, dict[int, RunData]]) -> None:
    kappas = []
    for tag, fold_data in data.items():
        org_kappa = (
            pl.from_pandas(get_original(fold_data, tag))
            .rename({"value": "kappa"})
            .drop("fold")
        )

        kappas.append(org_kappa)

    kappas_df: pl.DataFrame = pl.concat(kappas, how="vertical")
    kappa = kappas_df.unique(subset=["record", "dataset", "method"])
    _ = kappa.group_by("dataset").agg(
        pl.col("kappa").mean().alias("mean_kappa"),
        pl.col("kappa").std().alias("std_kappa"),
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=kappa,
        x="dataset",
        y="kappa",
        palette="pastel",
        showfliers=False,
    )

    sns.stripplot(
        data=kappa,
        x="dataset",
        y="kappa",
        color="black",
        jitter=True,
        size=4,
        alpha=0.6,
    )

    sns.pointplot(
        data=kappa,
        x="dataset",
        y="kappa",
        color="red",
        markers="x",
        markersize=8,
        err_kws={"linewidth": 0},
        scale=0.7,
    )

    # pastel = iter(sns.color_palette("pastel"))
    # handles = [
    #     plt.Line2D(  # type: ignore
    #         [0],
    #         [0],
    #         marker="x",
    #         color=next(pastel),
    #         markersize=10,
    #         linewidth=0,
    #         label=f"{datasets['mean_kappa']:.2f}",
    #     )
    #     for datasets in kappa_info.iter_rows(named=True)
    # ]

    # Formatting
    plt.xlabel("Dataset")
    plt.ylabel("Kappa")
    plt.title("Kappa of Pretrained Model")
    # plt.legend(
    #     handles=handles,
    #     title="Dataset mean",
    #     loc="upper right",
    #     fontsize=10,
    #     title_fontsize=12,
    # )
    plt.savefig("fig/pretrained_dataset_kappa.png")
    pass


if __name__ == "__main__":
    tags = ["1wHtRWs0", "1aYTnFXcM"]
    data = {tag: get_tag_data(tag) for tag in tags}

    pretrained_kappa_performance_plot(data)
    pass
