from abc import ABC, abstractmethod

from lightning import LightningModule

from src.config.config import ModelConfig


class ModelLoader(ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

    @abstractmethod
    def load_pretrained(self) -> LightningModule:
        pass
