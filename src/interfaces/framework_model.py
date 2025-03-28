from abc import ABC, abstractmethod
from typing import Any, Sequence

from lightning import LightningModule
from lightning.pytorch.loggers import CSVLogger, NeptuneLogger
from torch import Tensor


class FrameworkModel(LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.parameter_count: dict[str, dict | float | int] = {}

    def on_fit_start(self) -> None:
        if isinstance(self.logger, CSVLogger):
            print(self.parameter_count)
        elif isinstance(self.logger, NeptuneLogger):
            self.logger.experiment["model/parameter_count"] = self.parameter_count
            self.logger.experiment["model/config"] = getattr(self, "config")
            self.logger.experiment["sys/tags"].add(getattr(self, "experiment_id"))
        return None

    @abstractmethod
    def predict_step(
        self, batch: Any, batch_index: int
    ) -> tuple[Tensor, Tensor, Sequence[Any]]:
        pass

    @abstractmethod
    def loss(self, prediction: Tensor, true: Tensor) -> Tensor:
        pass

    @abstractmethod
    def measurements(self, prediction: Tensor, true: Tensor) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def is_classification_parameter(self, parameter_name: str) -> bool:
        pass
