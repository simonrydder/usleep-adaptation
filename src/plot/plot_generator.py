from src.plot.delta_kappa_vs_methods import (
    plot_delta_kappa_vs_methods,
)
from src.plot.delta_kappa_vs_parameters import (
    plot_delta_kappa_vs_parameters,
)
from src.plot.val_kappa import (
    plot_validation_kappa,
)
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import MethodData


def generate_plots(data: list[MethodData]) -> None:
    plot_validation_kappa(data, "eesm19")
    plot_delta_kappa_vs_methods(data, "eesm19")
    plot_delta_kappa_vs_parameters(data, "eesm19")


if __name__ == "__main__":
    data = load_data(
        datasets=["eesm19"], ids=[2]
    )  # Load saved data from results folder
    # data = get_data("eesm19")  # Get data from neptune
    generate_plots(data)
    pass
