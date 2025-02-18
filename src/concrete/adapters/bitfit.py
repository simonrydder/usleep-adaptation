from lightning import LightningModule

from src.interfaces.adapter import Adapter


class BitFit(Adapter):
    def __init__(self) -> None:
        super().__init__()

    def adapt(self, model: LightningModule) -> LightningModule:
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                param.requires_grad = True

        return model
