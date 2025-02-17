from copy import deepcopy

from lightning import LightningModule

from src.interfaces.model_updater import ModelUpdater


class StandardModelUpdater(ModelUpdater):
    def __init__(self) -> None:
        super().__init__()

    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        new_model = deepcopy(model)
        for name, param in new_model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return new_model
