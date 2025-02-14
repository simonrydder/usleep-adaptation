from abc import ABC, abstractmethod

from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader


class FineTuner(ABC):
    def __init__(self, trainer: Trainer) -> None:
        super().__init__()
        self.trainer = trainer

    @abstractmethod
    def train(self, model: LightningModule, train: DataLoader, val: DataLoader):
        pass

    @abstractmethod
    def test(self, model: LightningModule, test: DataLoader):
        pass
