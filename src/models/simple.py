import torch.nn as nn
from lightning import LightningModule


class Encoder(nn.Module): ...


class Simple(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
