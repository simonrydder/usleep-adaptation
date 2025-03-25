from typing import Any, Sequence

import torch.nn as nn
from torch import Tensor, optim

from src.interfaces.framework_model import FrameworkModel


class SimpleLinearModel(FrameworkModel):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x: Tensor):
        y = self.linear(x)
        return y

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        pred = self(x)

        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        pred = self(x)

        loss = nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        pred = self(x)

        loss = nn.functional.mse_loss(pred, y)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.01)

    def predict_step(
        self, batch: Any, batch_index: int
    ) -> tuple[Tensor, Tensor, Sequence[Any]]:
        return super().predict_step(batch, batch_index)

    def loss(self, prediction: Tensor, true: Tensor) -> Tensor:
        loss = nn.functional.mse_loss(prediction, true)
        return loss

    def measurements(self, prediction: Tensor, true: Tensor) -> dict[str, Tensor]:
        return {"loss": self.loss(prediction, true)}

    def is_classification_parameter(self, parameter_name: str) -> bool:
        return False


if __name__ == "__main__":
    SimpleLinearModel()
