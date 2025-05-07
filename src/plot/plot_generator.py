from src.plot.delta_kappa_vs_parameters import plot_delta_kappa_vs_parameters
from src.plot.kappa_vs_methods import plot_kappa_vs_methods
from src.plot.pretrained_kappa import plot_pretrained_kappa_performance
from src.plot.val_kappa import (
    plot_validation_kappa,
)


def generate_plots() -> None:
    plot_validation_kappa()
    plot_kappa_vs_methods()
    plot_pretrained_kappa_performance()
    plot_delta_kappa_vs_parameters()

    # plot_delta_kappa_vs_methods(data, "eesm19")


if __name__ == "__main__":
    # download_data(datasets=["eesm19", "isruc_sg2", "isruc_sg3", ""])
    generate_plots()
    pass
