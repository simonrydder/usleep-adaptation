import os
from typing import Literal

from lightning import Callback
from pydantic import BaseModel

from src.config._early_stopping_setting import EarlyStoppingSetting
from src.config.utils import load_yaml_content
from src.utils.callbacks import early_stopping, learning_rate_monitor, timer


class TrainerConfig(BaseModel):
    max_epochs: int
    accelerator: Literal["cpu", "gpu"]
    early_stopping: EarlyStoppingSetting | None = None
    learning_rate_monitor: Literal["epoch", "step"] | None = None
    timer: bool = False

    def get_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []

        if self.early_stopping is not None:
            callback = early_stopping(**self.early_stopping.model_dump())
            callbacks.append(callback)

        if self.learning_rate_monitor is not None:
            callback = learning_rate_monitor(
                logging_interval=self.learning_rate_monitor
            )
            callbacks.append(callback)

        if self.timer:
            callbacks.append(timer())

        return callbacks


def get_trainer_config(file: str) -> TrainerConfig:
    trainer_config_file = os.path.join("trainer", file)
    content = load_yaml_content(trainer_config_file)
    return TrainerConfig(**content)


if __name__ == "__main__":
    res = get_trainer_config("usleep")
    res = get_trainer_config("custom")
    pass
