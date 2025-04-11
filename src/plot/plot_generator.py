from src.plot.delta_kappa_vs_parameter_count import (
    plot_delta_kappas_vs_parameter_percentage,
)
from src.plot.pretrained_kappa import pretrained_kappa_performance_plot
from src.plot.real_delta_kappa import plot_delta_kappas_vs_methods
from src.plot.utils.neptune_api import get_tag_data
from src.plot.val_kappa import generate_validation_delta_kappa_plot


def main():
    tags = ["lLeqTtsL", "CZQINJTk"]
    data = {tag: get_tag_data(tag) for tag in tags}

    generate_validation_delta_kappa_plot(data)
    pretrained_kappa_performance_plot(data)
    plot_delta_kappas_vs_parameter_percentage(data)
    plot_delta_kappas_vs_methods(data)


if __name__ == "__main__":
    main()
    pass
