import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from src.plot.utils.neptune_api import (
    RunData,
    convert_to_polars,
    get_original,
    get_tag_data,
)


def generate_validation_delta_kappa_plot(data: dict[str, dict[int, RunData]]) -> None:
    plot_data = []
    for tag, folds in data.items():
        org_kappa = pl.from_pandas(get_original(folds, tag))
        for fold, run_data in folds.items():
            delta_kappa = get_delta_kappa_step(run_data, org_kappa)

            delta_kappa = delta_kappa.with_columns(
                pl.lit(tag).alias("tag"),
                pl.lit(fold).alias("fold"),
                pl.lit(run_data.config.experiment.dataset).alias("dataset"),
                pl.lit(run_data.config.experiment.method).alias("method"),
            )
            plot_data.append(delta_kappa)

    df = pl.concat(plot_data, how="vertical")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="epoch", y="delta_kappa", hue="method", markers=True, dashes=False
    )

    # Formatting
    plt.xlabel("Epoch")
    plt.ylabel("Delta Kappa")
    plt.title("Validation Delta Kappa by Epoch")
    plt.legend(title="Methods")
    plt.grid(True)

    # Show plot
    plt.savefig("fig/val_delta_kappa.png")


def get_delta_kappa_step(
    run_data: RunData,
    org_kappa: pl.DataFrame,
) -> pl.DataFrame:
    kappa_step = convert_to_polars(run_data.new_validation.val.kappa_step)

    kappa_records = convert_to_polars(run_data.new_validation.records)

    kappa = kappa_step.join(kappa_records, on="step").with_columns(
        (pl.col("step") // pl.col("record").n_unique()).alias("epoch")
    )

    kappa = kappa.join(
        org_kappa.select(pl.col("record"), pl.col("value").alias("org_kappa")),
        on="record",
        how="left",
    )
    delta_kappa = kappa.with_columns(
        (pl.col("old_kappa") - pl.col("value")).alias("delta_kappa")
    )

    return delta_kappa


if __name__ == "__main__":
    tags = ["1wHtRWs0", "1aYTnFXcM"]
    data = {tag: get_tag_data(tag) for tag in tags}
    generate_validation_delta_kappa_plot(data)
