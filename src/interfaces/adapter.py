from abc import ABC, abstractmethod

from src.config._adapter_config import AdapterMethodConfig
from src.interfaces.framework_model import FrameworkModel


class Adapter(ABC):
    def __init__(self, config: AdapterMethodConfig) -> None:
        super().__init__()

    @abstractmethod
    def adapt(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        pass
