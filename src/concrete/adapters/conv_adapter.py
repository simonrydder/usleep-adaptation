import torch.nn as nn
from lightning import LightningModule

from src.interfaces.adapter import Adapter


class ConvAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__()

    def adapt(self, model: LightningModule) -> LightningModule:
        return super().adapt(model)


class Conv1dAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: int,
        kernel: int,
        activation_function,
        **conv_kwargs,
    ) -> None:
        super().__init__()

        hidden_dim = in_channels // gamma

        self.depthwise_conv = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=kernel, **conv_kwargs
        )
        self.activation = activation_function
        self.pointwise_conv = nn.Conv1d(
            hidden_dim, out_channels, kernel_size=kernel, **conv_kwargs
        )
