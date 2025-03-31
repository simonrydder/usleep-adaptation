from src.plot.utils.neptune_api import get_tag_data
from src.plot.val_kappa import generate_validation_delta_kappa_plot


def main():
    tags = ["1wHtRWs0", "1aYTnFXcM"]
    data = {tag: get_tag_data(tag) for tag in tags}

    generate_validation_delta_kappa_plot(data)


if __name__ == "__main__":
    main()
