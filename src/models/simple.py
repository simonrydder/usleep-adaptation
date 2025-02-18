import torch.nn as nn
from lightning import LightningModule
from torch import Tensor, optim


class Simple(LightningModule):
    def __init__(self) -> None:
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
