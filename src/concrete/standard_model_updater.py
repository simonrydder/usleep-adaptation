from copy import deepcopy

from lightning import LightningModule

from src.interfaces.adapter import Adapter
from src.interfaces.model_updater import ModelUpdater


class StandardModelUpdater(ModelUpdater):
    def __init__(self, adapter: Adapter) -> None:
        super().__init__(adapter)
        self.adapter = adapter

    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        new_model = deepcopy(model)
        new_model = self._freeze_all_parameters(new_model)

        self.adapter.adapt(new_model)

        return new_model

    def _freeze_all_parameters(self, model: LightningModule) -> LightningModule:
        for param in model.parameters():
            param.requires_grad = False

        return model
