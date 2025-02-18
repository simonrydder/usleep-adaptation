import torch.nn as nn
from lightning import LightningModule

from src.interfaces.adapter import Adapter


class BatchNormAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__()

    def adapt(self, model: LightningModule) -> LightningModule:
        # Unfreeze only the BN layers.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Optionally, force the BN layers into train mode so their running stats update.
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

        return model
