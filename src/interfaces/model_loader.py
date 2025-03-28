from abc import ABC, abstractmethod

from src.config.config import Config, ModelConfig
from src.interfaces.framework_model import FrameworkModel


class ModelLoader(ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

    @abstractmethod
    def load_pretrained(self, config: Config) -> FrameworkModel:
        pass
