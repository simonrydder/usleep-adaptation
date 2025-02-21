from abc import ABC, abstractmethod

from lightning import LightningModule

from src.interfaces.adapter import Adapter


class ModelUpdater(ABC):
    def __init__(self, adapter: Adapter) -> None:
        super().__init__()

    @abstractmethod
    def adapt(self, model: LightningModule) -> LightningModule:
        pass
