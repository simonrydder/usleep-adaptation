from abc import ABC, abstractmethod

from lightning import LightningModule


class AdapterMethod(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, model: LightningModule) -> LightningModule:
        pass
