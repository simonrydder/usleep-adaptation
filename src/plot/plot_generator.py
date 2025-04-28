from src.plot.kappa_vs_methods import plot_kappa_vs_methods
from src.plot.pretrained_kappa import plot_pretrained_kappa_performance
from src.plot.val_kappa import (
    plot_validation_kappa,
)


def generate_plots() -> None:
    plot_validation_kappa()
    plot_kappa_vs_methods()
    plot_pretrained_kappa_performance()

    # plot_delta_kappa_vs_methods(data, "eesm19")
    # plot_delta_kappa_vs_parameters(data, "eesm19")


if __name__ == "__main__":
    generate_plots()
    pass
