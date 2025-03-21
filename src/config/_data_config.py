import os
from typing import Literal

from pydantic import BaseModel

from src.config.utils import load_yaml_content


class DataConfig(BaseModel):
    dataset: str
    type: Literal["custom", "hdf5"]
    batch_size: int
    sleep_epochs: int = 35
    num_batches: int = 3
    num_workers: int | tuple[int, int, int]
    split_percentages: tuple[float, float, float]
    random_state: int
    num_samples: int | None = None


def get_data_config(file: str) -> DataConfig:
    data_config_file = os.path.join("dataset", file)
    yaml_content = load_yaml_content(data_config_file)
    return DataConfig(**yaml_content)


if __name__ == "__main__":
    content = load_yaml_content("dataset/eesm19")
    res = get_data_config("eesm19")
    pass
