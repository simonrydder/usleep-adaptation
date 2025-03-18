from abc import ABC, abstractmethod

from lightning import LightningModule

from src.config._adapter_config import AdapterMethodConfig


class Adapter(ABC):
    def __init__(self, config: AdapterMethodConfig) -> None:
        super().__init__()

    @abstractmethod
    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        pass
