import lightning as pl  # or from lightning import LightningModule if using that package
import torch.nn as nn
from torch import Tensor, optim
from torchvision.models import resnet18


class Resnet(pl.LightningModule):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Load ResNet-18
        self.model = resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.cross_entropy(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.cross_entropy(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = nn.functional.cross_entropy(pred, y)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.001)
