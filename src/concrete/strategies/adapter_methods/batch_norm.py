import torch.nn as nn
from lightning import LightningModule

from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.adapter_method import AdapterMethod


class BatchNorm(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        for name, child_module in model.named_children():
            setattr(model, name, self.recursive_apply(child_module))

        return model

    def recursive_apply(self, parent: nn.Module | LightningModule) -> nn.Module:
        if isinstance(parent, nn.BatchNorm1d):
            for param in parent.parameters():
                param.requires_grad = True

        if isinstance(parent, nn.BatchNorm2d):
            for param in parent.parameters():
                param.requires_grad = True

        for name, child in parent.named_children():
            setattr(parent, name, self.recursive_apply(child))

        return parent
