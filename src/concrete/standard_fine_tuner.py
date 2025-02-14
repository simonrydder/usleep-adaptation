from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from src.interfaces.fine_tuner import FineTuner


class StandardFineTuner(FineTuner):
    def __init__(self, trainer: Trainer) -> None:
        super().__init__(trainer)

    def train(self, model: LightningModule, train: DataLoader, val: DataLoader):
        return self.trainer.fit(model, train, val)

    def test(self, model: LightningModule, test: DataLoader):
        _ = self.trainer.test(model, test)
