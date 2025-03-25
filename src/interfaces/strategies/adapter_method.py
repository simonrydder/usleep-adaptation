from abc import ABC, abstractmethod

from src.interfaces.framework_model import FrameworkModel


class AdapterMethod(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        pass
