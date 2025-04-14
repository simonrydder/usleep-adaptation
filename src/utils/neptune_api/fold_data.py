from neptune import Run
from pydantic import BaseModel

from src.utils.neptune_api.config_data import ConfigData, get_config_data
from src.utils.neptune_api.parameter_data import ParameterData, get_parameter_data
from src.utils.neptune_api.performance_data import PerformanceData, get_performance_data


class FoldData(BaseModel):
    config: ConfigData
    parameters: ParameterData
    validation_step: list[PerformanceData]
    train_step: list[PerformanceData]


def get_fold_data(run: Run) -> FoldData:
    return FoldData(
        config=get_config_data(run),
        parameters=get_parameter_data(run),
        validation_step=get_performance_data(run, "new", "val"),
        train_step=get_performance_data(run, "new", "train"),
    )
