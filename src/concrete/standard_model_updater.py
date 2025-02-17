from copy import deepcopy

from lightning import LightningModule

from src.interfaces.adapter import Adapter
from src.interfaces.model_updater import ModelUpdater


class StandardModelUpdater(ModelUpdater):
    def __init__(self, adapter: type[Adapter]) -> None:
        super().__init__(adapter)
        self.adapter = adapter()

    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        new_model = deepcopy(model)
        self.adapter.adapt(new_model)

        return new_model
