from abc import ABC, abstractmethod

from lightning import LightningModule

from src.interfaces.strategies.adapter_method import AdapterMethod


class Adapter(ABC):
    def __init__(self, adapter_method: AdapterMethod) -> None:
        super().__init__()

    @abstractmethod
    def adapt(self, model: LightningModule) -> LightningModule:
        pass
