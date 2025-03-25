from abc import ABC, abstractmethod

from src.config.config import ModelConfig
from src.config.experiment import Experiment
from src.interfaces.framework_model import FrameworkModel


class ModelLoader(ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

    @abstractmethod
    def load_pretrained(self, experiment: Experiment) -> FrameworkModel:
        pass
