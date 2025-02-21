import torch
import torch.nn as nn
from lightning import LightningModule

from src.concrete.strategies.conv_adapter_creators._conv2dadapter import Conv2dAdapter
from src.interfaces.adapter import Adapter
from src.interfaces.strategies.forward_pass import ForwardPass


class ConvAdapter(Adapter):
    def __init__(self, forward_pass: ForwardPass, residual: bool) -> None:
        super().__init__()
        self.forward_pass = forward_pass
        self.residual = residual

    def adapt(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_adapt(child_module))

        return model

    def recursive_adapt(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, nn.Conv2d):
            return self.create_new_module(parent)

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_adapt(child))

        return parent

    def create_new_module(self, module: nn.Conv2d) -> nn.Module:
        adapter = Conv2dAdapter(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel=module.kernel_size,  # type: ignore
            gamma=4,
            activation_function=nn.ReLU,
            stride=module.stride,
            dilation=module.dilation,
            padding=module.padding,
            bias=module.bias,
            groups=module.groups,
        )
        new_module = ConvAdapterModule(module, adapter, self.forward_pass)

        return new_module


class ConvAdapterModule(nn.Module):
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
