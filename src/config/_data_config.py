import os
from typing import Literal

from pydantic import BaseModel, field_serializer

from src.config.utils import load_yaml_content


class DataConfig(BaseModel):
    dataset: str
    type: Literal["custom", "hdf5"]
    batch_size: int
    sleep_epochs: int
    num_batches: int
    num_workers: int | tuple[int, int, int]
    train_size: int | None
    validation_size: int
    num_fold: int
    num_samples: int | None = None

    # @field_serializer("split_percentages")
    # def serialize_split(self, v):
    #     return {"train": v[0], "val": v[1], "test": v[2]}

    @field_serializer("num_workers")
    def serialize_num_workers(self, v):
        if isinstance(v, int):
            v = (v, v, v)

        return {"train": v[0], "val": v[1], "test": v[2]}


def get_data_config(file: str) -> DataConfig:
    data_config_file = os.path.join("dataset", file)
    yaml_content = load_yaml_content(data_config_file)
    return DataConfig(**yaml_content)


if __name__ == "__main__":
    content = load_yaml_content("dataset/eesm19")
    res = get_data_config("eesm19")
    pass
