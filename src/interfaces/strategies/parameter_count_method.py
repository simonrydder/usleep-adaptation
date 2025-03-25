from abc import ABC, abstractmethod

from src.interfaces.framework_model import FrameworkModel


class ParameterCountMethod(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_parameter_count(self, model: FrameworkModel) -> FrameworkModel:
        pass
