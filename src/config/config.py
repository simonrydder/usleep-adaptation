from pydantic import BaseModel

from src.config._adapter_config import (
    AdapterMethodConfig,
    get_adapter_method_config,
)
from src.config._data_config import DataConfig, get_data_config
from src.config._model_config import ModelConfig, get_model_config
from src.config._trainer_config import TrainerConfig, get_trainer_config
from src.config.experiment import Experiment


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    adapter: AdapterMethodConfig
    trainer: TrainerConfig


def load_config(experiment: Experiment) -> Config:
    return Config(
        model=get_model_config(experiment.model),
        data=get_data_config(experiment.dataset),
        adapter=get_adapter_method_config(experiment.method),
        trainer=get_trainer_config(experiment.trainer),
    )


if __name__ == "__main__":
    conf = load_config(
        Experiment(
            dataset="eesm19",
            method="bitfit",
            model="usleep",
            trainer="usleep",
        )
    )

    pass
