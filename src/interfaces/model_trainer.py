from abc import ABC, abstractmethod

from lightning import Trainer

from src.config._trainer_config import TrainerConfig
from src.config.experiment import Experiment


class ModelTrainer(ABC):
    def __init__(self, config: TrainerConfig, experiment: Experiment) -> None:
        super().__init__()

    @abstractmethod
    def get(self) -> Trainer:
        pass
