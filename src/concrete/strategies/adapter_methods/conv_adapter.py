from typing import Type

import torch
import torch.nn as nn
from lightning import LightningModule
from pydantic import BaseModel

from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.forward_pass import ForwardPass


class ConvolutionSetting(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int] | tuple[int, int]
    stride: int | tuple[int] | tuple[int, int]
    padding: int | tuple[int] | tuple[int, int] | str
    dilation: int | tuple[int] | tuple[int, int]
    groups: int
    bias: bool


class ConvAdapter(AdapterMethod):
    def __init__(
        self,
        forward_pass: ForwardPass,
        activation: Type[nn.Module],
        reduction: int | None | None = None,
        kernel: int | tuple[int] | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.forward_pass = forward_pass
        self.reduction = reduction
        self.activation = activation
        self.kernel = kernel

    def apply(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, nn.Conv1d):
            return self.create_adapter_module(parent)

        if isinstance(parent, nn.Conv2d):
            return self.create_adapter_module(parent)

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent

    def create_adapter_module(self, module: nn.Conv2d | nn.Conv1d) -> nn.Module:
        adapter = self.create_adapter(module)

        return ConvAdapterModule(module, adapter, self.forward_pass)

    def create_adapter(self, module: nn.Conv2d | nn.Conv1d):
        setting = self.create_conv_setting(module)
        adapter = ConvAdapterBlock(
            self.reduction, self.kernel, self.activation, type(module), setting
        )

        return adapter

    def create_conv_setting(self, module: nn.Conv2d | nn.Conv1d) -> ConvolutionSetting:
        return ConvolutionSetting(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,  # type: ignore
            stride=module.stride,  # type: ignore
            dilation=module.dilation,  # type: ignore
            padding=module.padding,  # type: ignore
            groups=module.groups,
            bias=module.bias is not None,
        )


class ConvAdapterBlock(nn.Module):
    def __init__(
        self,
        reduction: int | None,
        kernel: int | tuple[int] | tuple[int, int] | None,
        activation_function: Type[nn.Module],
        conv_type: Type[nn.Conv1d] | Type[nn.Conv2d],
        conv_settings: ConvolutionSetting,
    ) -> None:
        super().__init__()
        in_channels = conv_settings.in_channels
        out_channels = conv_settings.out_channels
        del conv_settings.in_channels
        del conv_settings.out_channels

        if reduction is None:
            reduction = in_channels  # -> hidden dim = 1

        hidden_dim = max(in_channels // reduction, 1)

        if kernel is not None:
            conv_settings.kernel_size = kernel

        # Depth-wise Convolution
        self.W_down = conv_type(in_channels, hidden_dim, **conv_settings.model_dump())

        # Activation function
        self.activation = activation_function()

        # Point-wise Convolution: kernel_size = 1
        self.W_up = conv_type(hidden_dim, out_channels, kernel_size=1)

        # Alpha parameter for scaling output
        alpha_shape = (1, out_channels, 1)
        if conv_type == nn.Conv2d:
            alpha_shape = (*alpha_shape, 1)

        self.alpha_scale = nn.Parameter(torch.ones(alpha_shape), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W_down(x)
        h = self.activation(h)
        h = self.W_up(h)

        return h * self.alpha_scale  # Optional if alpha scaling is unnecessary


class ConvAdapterModule(LightningModule):
    def __init__(
        self,
        original: nn.Module,
        adapter: nn.Module,
        forward_pass: ForwardPass,
    ) -> None:
        super().__init__()

        self.original = original
        self.adapter = adapter
        self.forward_pass = forward_pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pass.forward(
            x, original=self.original, adapter=self.adapter
        )
