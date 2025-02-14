from abc import ABC, abstractmethod

from lightning import LightningModule


class ModelUpdater(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def adapt(self, model: LightningModule) -> LightningModule:
        pass
