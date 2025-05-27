from src.plot.convergence_time import convergence_plotting
from src.plot.delta_kappa_table import delta_kappa_table
from src.plot.delta_kappa_vs_parameters import plot_delta_kappa_vs_parameters
from src.plot.kappa_vs_methods import plot_kappa_vs_methods
from src.plot.new_delta_kappa_vs_training_size import new_plot_delta_kappa_vs_train_size
from src.plot.pretrained_kappa import plot_pretrained_kappa_performance
from src.plot.val_kappa import (
    plot_validation_kappa,
)


def generate_plots() -> None:
    # Pretrained model performance
    plot_pretrained_kappa_performance()

    # Convergence time + Appendix
    convergence_plotting()
    plot_validation_kappa()

    # Fine-tuned performnace
    print(delta_kappa_table())
    plot_kappa_vs_methods()

    # Performance based on number of parameters
    # print(parameter_scaled_delta_kappa_table())
    plot_delta_kappa_vs_parameters()

    # Performance based on dataset size
    new_plot_delta_kappa_vs_train_size()
    # plot_delta_kappa_vs_train_size()

    # plot_delta_kappa_vs_methods(data, "eesm19")


if __name__ == "__main__":
    # download_data(datasets=["eesm19", "isruc_sg2", "isruc_sg3", ""])
    generate_plots()
    pass
