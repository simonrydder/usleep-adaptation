import torch
import torch.nn as nn
from lightning import LightningModule

from src.concrete.strategies.conv_adapter_creators._conv1dadapter import Conv1dAdapter
from src.concrete.strategies.conv_adapter_creators._conv2dadapter import Conv2dAdapter
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.forward_pass import ForwardPass


class ConvAdapter(AdapterMethod):
    def __init__(self, forward_pass: ForwardPass, reduction: int | None) -> None:
        super().__init__()
        self.forward_pass = forward_pass
        self.gamma = reduction
        self.activation = nn.ReLU
        self.kernel = None

    def apply(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        adapter = self.create_adapter(parent)

        if adapter is not None:
            return ConvAdapterModule(parent, adapter, self.forward_pass)

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent

    def create_adapter(self, module: nn.Module | LightningModule) -> nn.Module | None:
        if isinstance(module, nn.Conv2d):
            return Conv2dAdapter(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel=module.kernel_size,  # type: ignore
                gamma=module.in_channels if self.gamma is None else self.gamma,
                activation_function=self.activation,
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                bias=module.bias,
                groups=module.groups,
            )

        if isinstance(module, nn.Conv1d):
            return Conv1dAdapter(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel=module.kernel_size,  # type: ignore
                gamma=module.in_channels if self.gamma is None else self.gamma,
                activation_function=self.activation,
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                bias=True if module.bias is not None else False,
                groups=module.groups,
            )

        return None


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
