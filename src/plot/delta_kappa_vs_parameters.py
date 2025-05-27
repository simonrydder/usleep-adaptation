import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter
from scipy.stats import linregress

from src.plot.colors import BASE_COLOR, HIGHLIGHT_COLOR
from src.utils.figures import adjust_axis_font, save_figure
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import (
    extract_performance,
    extract_settings,
)


def plot_delta_kappa_vs_parameters(show: bool = False) -> None:
    data = _get_delta_kappa_data()
    _plot_delta_kappe_vs_parameters_separate_datasets(
        data, log=False, col_wrap=3, show=show
    )
    _plot_delta_kappe_vs_parameters_separate_datasets(
        data, log=True, col_wrap=3, show=show
    )
    _plot_delta_kappa_vs_parameters_joined_datasets(data, log=True, show=show)
    _plot_delta_kappa_vs_parameters_joined_datasets(data, log=False, show=show)


def _get_delta_kappa_data() -> pl.DataFrame:
    raw_data = load_data()

    dfs = []
    for method_data in raw_data:
        if method_data.train_size is not None:
            continue

        test = extract_performance(method_data, "new").drop("accuracy", "loss", "f1")
        base = extract_performance(method_data, "org").drop("accuracy", "loss", "f1")

        delta_kappa = test.join(
            base, on=["method", "record", "dataset", "key"], how="inner", suffix="_base"
        ).with_columns((pl.col("kappa") - pl.col("kappa_base")).alias("delta_kappa"))

        parameters = (
            extract_settings(method_data)
            .select("dataset", "key", "method", "free_parameters")
            .unique(keep="any")
        )
        assert len(parameters) == 1, "Parameters should be the same for all folds"

        params_delta_kappa = delta_kappa.join(
            parameters, on=["dataset", "method", "key"], how="left"
        )
        dfs.append(params_delta_kappa)

    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    selected = df.select(
        "dataset",
        "method",
        "key",
        "kappa",
        "delta_kappa",
        "free_parameters",
    )
    delta_kappa_mean = selected.group_by(["dataset", "method", "key"]).agg(
        pl.col("kappa").mean(),
        pl.col("delta_kappa").mean(),
        pl.col("free_parameters").mean(),
    )

    return delta_kappa_mean.sort("dataset", "method")


def _plot_delta_kappa_vs_parameters(
    data: pl.DataFrame | pd.DataFrame,
    log: bool,
    full_params: int,
    color=None,
    g: sns.FacetGrid | None = None,
    col_wrap: int | None = None,
):
    ax = plt.gca()

    if isinstance(data, pd.DataFrame):
        dataset: str = data.iloc[0, 0]  # type: ignore
    else:
        dataset: str = data.item(0, "dataset")

    sns.scatterplot(
        data=data,
        x="free_parameters",
        y="delta_kappa",
        color=HIGHLIGHT_COLOR[dataset],
        alpha=0.6,
        ax=ax,
    )

    ax.set_xlabel("Number of Free Parameters", fontsize=12)
    ax.set_ylabel("Delta Kappa", fontsize=12)

    def params_to_percent(x):
        return (x / full_params) * 100

    def percent_to_params(p):
        return (p / 100) * full_params

    if log:
        ax.set_xscale("log")
        ax.set_xlim(left=10)
        ax.xaxis.set_minor_locator(LogLocator(subs="all", base=10.0))
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if col_wrap is not None and g is not None:
        ax_index = np.where(g.axes == ax)[0][0]
        is_top_row = ax_index < col_wrap
    else:
        is_top_row = True

    secax = ax.secondary_xaxis("top", functions=(params_to_percent, percent_to_params))
    secax.set_xlabel("% of Free Parameters" if is_top_row else "", fontsize=12)
    secax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    adjust_axis_font(ax.xaxis, size=12)
    adjust_axis_font(ax.yaxis, size=12)
    adjust_axis_font(secax.xaxis, size=12)
    ax.spines["right"].set_visible(True)
    ax.grid(True)


def _plot_delta_kappa_vs_parameters_joined_datasets(
    data: pl.DataFrame, log: bool, show: bool = False
) -> None:
    # sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(10, 6))
    full_params = data.filter(pl.col("method") == "Full").item(0, "free_parameters")
    _plot_delta_kappa_vs_parameters(data, log=log, full_params=full_params)

    fig.suptitle("Delta Kappa vs. Number of Free Parameters", size=14)
    plt.legend(fontsize=10, title="Dataset")
    plt.grid(True)
    # plt.tight_layout()

    plt.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.1)

    if show:
        plt.show()

    file_dst = f"figures/delta_kappa_vs_parameters_joined{'_log' if log else ''}.png"
    save_figure(fig, file_dst)

    return None


def _plot_delta_kappe_vs_parameters_separate_datasets(
    data: pl.DataFrame, log: bool, col_wrap: int, show: bool
) -> None:
    full_params = data.filter(pl.col("method") == "Full").item(0, "free_parameters")

    g = sns.FacetGrid(
        data,
        col="dataset",
        sharex=True,
        sharey=True,
        height=4,
        aspect=1.5,
        col_wrap=col_wrap,
    )

    g.map_dataframe(
        _plot_delta_kappa_vs_parameters,
        log=log,
        full_params=full_params,
        g=g,
        col_wrap=col_wrap,
    )

    g.set_titles(template="{col_name}", size=13)
    g.figure.subplots_adjust(
        top=0.89,
        bottom=0.055,
        left=0.055,
        right=0.97,
        wspace=0.09,
        hspace=0.33,
    )
    g.figure.suptitle(
        "Delta Kappa vs. Number of Free Parameters by Dataset", fontsize=15
    )

    for ax in g.axes.flatten():
        for line in ax.get_ygridlines():
            if line.get_ydata()[0] == 0:
                # line.set_linewidth(1.5)  # or whatever width you prefer
                line.set_color("black")

    for ax, (_, df) in zip(g.axes.flatten(), data.group_by("dataset")):
        assert isinstance(ax, Axes)
        color = BASE_COLOR[df.item(0, "dataset").lower()]
        x = df.get_column("free_parameters").to_numpy()
        y = df.get_column("delta_kappa").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]

        slope, intercept, *_ = linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = intercept + slope * (x_fit)
        add = "-" if intercept < 0 else "+"  # type: ignore
        label = rf"$y = {slope * x.max():.3f}x {add} {abs(intercept):.3f}$"
        ax.plot(x_fit, y_fit, color=color, label=label, linewidth=2)
        ax.legend(loc="best", fontsize=9, frameon=True)
        pass

    save_figure(
        g.figure, f"figures/delta_kappa_vs_parameters{'_log' if log else ''}.png"
    )

    if show:
        plt.show()

    return None


if __name__ == "__main__":
    plot_delta_kappa_vs_parameters()
