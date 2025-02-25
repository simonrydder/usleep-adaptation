from abc import ABC, abstractmethod

from lightning import Trainer

from src.config._trainer_config import TrainerConfig


class ModelTrainer(ABC):
    def __init__(self, config: TrainerConfig) -> None:
        super().__init__()

    @abstractmethod
    def get(self) -> Trainer:
        pass
