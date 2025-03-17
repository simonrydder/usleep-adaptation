from lightning import LightningModule
from torch import nn

import src.concrete.strategies.adapter_methods.kronA_base as kron
from src.interfaces.strategies.adapter_method import AdapterMethod


class KronA(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, nn.Linear):
            kron_linear = self.create_new_module(parent)
            return kron_linear

        if isinstance(parent, nn.Conv1d):
            kron_conv1d = self.create_new_module(parent)
            return kron_conv1d

        if isinstance(parent, nn.Conv2d):
            kron_conv2d = self.create_new_module(parent)
            return kron_conv2d

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent

    def create_new_module(self, module: nn.Linear | nn.Conv1d | nn.Conv2d):
        if isinstance(module, nn.Conv1d):
            return kron.Conv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                bias=True if module.bias is not None else False,
                groups=module.groups,
                r=7,
                kron_alpha=1,
                kron_dropout=0,
            )

        elif isinstance(module, nn.Conv2d):
            return kron.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride,
                dilation=module.dilation,
                padding=module.padding,
                bias=True if module.bias is not None else False,
                groups=module.groups,
                r=7,
                kron_alpha=1,
                kron_dropout=0,
            )
        return kron.KronALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False,
            r=7,
            kron_alpha=1,
            kron_dropout=0,
        )
