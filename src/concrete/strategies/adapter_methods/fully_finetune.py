from lightning import LightningModule

from src.interfaces.strategies.adapter_method import AdapterMethod


class FullyFinetune(AdapterMethod):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: LightningModule) -> LightningModule:
        for param in model.parameters():
            param.requires_grad = True

        return model
