from src.plot.delta_kappa_vs_parameter_count import (
    plot_delta_kappas_vs_parameter,
)
from src.plot.pretrained_kappa import pretrained_kappa_performance_plot
from src.plot.real_delta_kappa import (
    plot_performance_delta_kappa_vs_methods,
)
from src.plot.val_kappa import (
    plot_validation_delta_kappa,
)
from src.utils.neptune_api.data_loader import load_data
from src.utils.neptune_api.method_data import MethodData


def generate_plots(data: dict[str, MethodData]) -> None:
    plot_validation_delta_kappa(data, "eesm19")
    plot_performance_delta_kappa_vs_methods(data, "eesm19")


def main():
    pretrained_kappa_performance_plot(data)
    plot_delta_kappas_vs_parameter(data)


if __name__ == "__main__":
    data = load_data("eesm19")  # Load saved data from results folder
    # data = get_data('eesm19') # Get data from neptune
    generate_plots(data)
    pass
