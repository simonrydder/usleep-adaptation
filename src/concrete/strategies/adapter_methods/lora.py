import loralib as lora
import torch.nn as nn
from lightning import LightningModule

from src.interfaces.strategies.adapter_method import AdapterMethod


class LoRA(AdapterMethod):
    def __init__(self, rank: int = 1, alpha: int = 1, dropout: float = 0.0) -> None:
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, nn.Linear):
            lora_linear = self.create_new_module(parent)
            return lora_linear

        if isinstance(parent, nn.Conv1d):
            lora_conv1d = self.create_new_module(parent)
            return lora_conv1d

        if isinstance(parent, nn.Conv2d):
            lora_conv2d = self.create_new_module(parent)
            return lora_conv2d

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent

    def create_new_module(self, module: nn.Linear | nn.Conv1d | nn.Conv2d):
        if isinstance(module, nn.Conv1d):
            return lora.ConvLoRA(
                lora.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride,
                    dilation=module.dilation,
                    padding=module.padding,
                    bias=True if module.bias is not None else False,
                    groups=module.groups,
                ),
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
            )

        elif isinstance(module, nn.Conv2d):
            return lora.ConvLoRA(
                lora.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride,
                    dilation=module.dilation,
                    padding=module.padding,
                    bias=True if module.bias is not None else False,
                    groups=module.groups,
                ),
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
            )
        return lora.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
        )
