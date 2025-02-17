from abc import ABC, abstractmethod

from lightning import Trainer


class ModelTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get(self) -> Trainer:
        pass
