from neptune import Run
from pydantic import BaseModel

from src.config.experiment import Experiment
from src.utils.neptune_api.neptune_api import get_data_scalar, get_run


class Sizes(BaseModel):
    train: int
    validation: int
    test: int


class DataConfig(BaseModel):
    batch_size: int
    num_batches: int
    num_fold: int
    sleep_epochs: int
    sizes: Sizes
    validation_size: int


class ConfigData(BaseModel):
    experiment: Experiment
    data: DataConfig


def get_config_data(run: Run) -> ConfigData:
    return ConfigData(**get_data_scalar(run, "model/config"))


if __name__ == "__main__":
    x = get_config_data(get_run("US-524"))
    pass
