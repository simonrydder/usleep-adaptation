from lightning import LightningModule

from src.interfaces.model_updater import ModelUpdater


class StandardModelUpdater(ModelUpdater):
    def __init__(self, adapter) -> None:
        super().__init__()

    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model
