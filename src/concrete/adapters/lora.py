import loralib
from lightning import LightningModule
from torch.nn import Conv2d, Linear

from src.interfaces.adapter import Adapter


class LoRA(Adapter):
    def __init__(self) -> None:
        super().__init__()

    def adapt(self, model: LightningModule) -> LightningModule:
        for block in model.modules():
            if isinstance(block, Conv2d):
                block = loralib.Conv2d(
                    block.in_channels,
                    block.out_channels,
                    block.kernel_size[0],
                    block.stride,
                    block.padding,
                    True if block.bias == None else False,
                )
            elif isinstance(block, Linear):
                block = loralib.Linear(
                    in_features=block.in_features,
                    out_features=block.out_features,
                    r=16,
                )

            else:
                pass

        return model
