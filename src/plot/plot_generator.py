from src.plot.delta_kappa_vs_methods import (
    plot_delta_kappa_vs_methods,
)
from src.plot.delta_kappa_vs_parameters import (
    plot_delta_kappa_vs_parameters,
)
from src.plot.kappa_vs_methods import plot_kappa_vs_methods
from src.plot.pretrained_kappa import pretrained_kappa_performance_plot
from src.plot.val_kappa import (
    plot_validation_kappa,
)
from src.utils.neptune_api.data_loader import get_data, load_data
from src.utils.neptune_api.method_data import MethodData


def generate_plots(data: list[MethodData]) -> None:
    plot_validation_kappa(data, "eesm19")
    plot_delta_kappa_vs_methods(data, "eesm19")
    plot_delta_kappa_vs_parameters(data, "eesm19")
    plot_kappa_vs_methods(data, "eesm19")

    pretrained_kappa_performance_plot(data)


if __name__ == "__main__":
    use_downloaded = True

    datasets = ["eesm19"]
    ids = [3]
    methods = None

    if use_downloaded:
        data = load_data(datasets, methods, ids)
    else:
        data = get_data(datasets, methods, ids)

    generate_plots(data)
    pass
