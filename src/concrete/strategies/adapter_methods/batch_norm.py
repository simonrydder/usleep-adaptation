import torch.nn as nn
from lightning import LightningModule

from src.interfaces.strategies.adapter_method import AdapterMethod


class BatchNorm(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        # TODO: FIX!
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Optionally, force the BN layers into train mode so their running stats update.
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

        return model
