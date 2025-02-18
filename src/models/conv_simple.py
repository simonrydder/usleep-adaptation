import torch.nn as nn
from lightning import LightningModule
from torch import Tensor, optim


class ConvSimple(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 56 * 56, 10)

    def forward(self, x: Tensor):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_index: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.01)
