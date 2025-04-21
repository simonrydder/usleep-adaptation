import itertools
import os

import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm

from src.concrete.standard_adapter import StandardAdapter
from src.concrete.standard_model_loader import StandardModelLoader
from src.config.config import Config, load_config
from src.config.experiment import Experiment
from src.interfaces.framework_model import FrameworkModel
from src.utils.decorators import suppress_prints


@suppress_prints
def create_model(config: Config) -> FrameworkModel:
    model_loader = StandardModelLoader(config.model)
    org_model = model_loader.load_pretrained(config)

    adapter = StandardAdapter(config.adapter)
    new_model = adapter.adapt(org_model)

    return new_model


def create_config(reduction: int | None, forward_pass: str, layer: bool) -> Config:
    exp = Experiment(dataset="eesm19", method="PCA3", model="usleep", trainer="usleep")
    config = load_config(exp)

    config.adapter.settings.reduction = reduction
    config.adapter.settings.forward_pass = forward_pass
    config.adapter.settings.layer = layer

    return config


def get_procentage(model: FrameworkModel) -> float:
    org_total = (
        model.parameter_count["model"]["frozen"]  # type: ignore
        + model.parameter_count["classification"]["free"]  # type: ignore
    )
    new_free = (
        model.parameter_count["model"]["free"]  # type: ignore
        + model.parameter_count["classification"]["free"]  # type: ignore
    )

    return round(new_free / org_total, 4)


def main(show: bool):
    df = raw_data()

    types = ["parallel", "sequential"]
    layers = [False, True]
    target_procentages = [0.05, 0.1, 0.2, 0.5]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    sns.set_theme(style="whitegrid", context="paper")

    axs = axs.flatten()
    for i, (layer, type_) in enumerate(itertools.product(layers, types)):
        ax = axs[i]
        assert isinstance(ax, Axes)
        df_pandas = df.filter(
            pl.col("type") == type_, pl.col("layer") == layer
        ).to_pandas()

        df_normal = df_pandas[df_pandas["reduction"].notnull()]
        df_none = df_pandas[df_pandas["reduction"].isnull()]

        # Plot normal lines
        ax.scatter(df_normal["reduction"], df_normal["procentage"], color="black")

        # Plot dashed line for reduction=None
        y_none = df_none["procentage"].values[0]
        ax.axhline(y=y_none, linestyle="--", color="black", linewidth=1.5)

        closest_points = []
        # Find and plot closest points for each target
        for target in target_procentages:
            df_normal["diff"] = (df_normal["procentage"] - target).abs()
            closest_row = df_normal.loc[df_normal["diff"].idxmin()]
            x, y = closest_row["reduction"], closest_row["procentage"]
            closest_points.append((x, y, target))
            ax.scatter(
                [x],
                [y],
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                s=100,
                zorder=6,
                marker="o",
                label=f"{int(target * 100)}% → ({int(x)}, {round(y, 4)})",  # type: ignore
            )

        # Title logic
        title = ("P" if type_ == "parallel" else "S") + "C" + ("L" if layer else "A")
        ax.set_title(title, fontdict={"size": 14})

        ax.set_xlabel("Reduction")
        ax.set_ylabel("Procentage")
        ax.grid()

        # Optional: avoid duplicate legends
        ax.legend()

    plt.tight_layout()
    plt.savefig("figures/conv_adapter_size.png")

    if show:
        plt.show()

    return None


def raw_data():
    file = os.path.join("data", "other", "conv_adapter_size.csv")
    if os.path.exists(file):
        return pl.read_csv(file)

    data = []
    for reduction in tqdm(list(range(1, 36)) + [None]):
        for forward_pass in ["sequential", "parallel"]:
            for layer in [True, False]:
                conf = create_config(reduction, forward_pass, layer)
                model = create_model(conf)
                proc = get_procentage(model)
                data.append(
                    {
                        "reduction": reduction,
                        "type": forward_pass,
                        "layer": layer,
                        "total": model.parameter_count["total"],
                        "model_free": model.parameter_count["model"]["free"],  # type: ignore
                        "procentage": proc,
                    }
                )

    df = pl.DataFrame(data)
    df.write_csv(file)

    return df


if __name__ == "__main__":
    main(False)
