from abc import ABC, abstractmethod

from lightning import LightningModule


class ParameterCountMethod(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_parameter_count(self, model: LightningModule) -> LightningModule:
        pass
