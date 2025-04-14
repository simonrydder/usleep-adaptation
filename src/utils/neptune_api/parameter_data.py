from neptune import Run
from pydantic import BaseModel

from src.utils.neptune_api.neptune_api import get_data_scalar, get_run


class TrainableParameterData(BaseModel):
    free: int = 0
    frozen: int = 0


class ParameterData(BaseModel):
    total: int
    classification: TrainableParameterData
    model: TrainableParameterData


def get_parameter_data(run: Run) -> ParameterData:
    return ParameterData(**get_data_scalar(run, "model/parameter_count"))


if __name__ == "__main__":
    x = get_parameter_data(get_run("US-524"))
    pass
