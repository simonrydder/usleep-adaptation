from typing import Type

import torch.nn as nn
from lightning import LightningModule

from csdp.ml_architectures.usleep.usleep import ConvBNELU
from src.interfaces.strategies.adapter_method import AdapterMethod
from src.interfaces.strategies.forward_pass import ForwardPass


class ResidualAdapter(AdapterMethod):
    def __init__(
        self, forward_pass: ForwardPass, activation: Type[nn.Module], reduction
    ) -> None:
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(
        self, parent: nn.Module | LightningModule, is_decoder_block: bool = False
    ) -> nn.Module:
        if isinstance(parent, ConvBNELU) and is_decoder_block:
            return self.create_adapter_module(parent)
        return parent
