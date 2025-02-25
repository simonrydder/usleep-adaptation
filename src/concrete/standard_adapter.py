from copy import deepcopy

from lightning import LightningModule

from src.interfaces.adapter import Adapter
from src.interfaces.strategies.adapter_method import AdapterMethod


class StandardAdapter(Adapter):
    def __init__(self, adapter_method: AdapterMethod) -> None:
        super().__init__(adapter_method)
        self.adapter_method = adapter_method

    def adapt(self, model: LightningModule) -> LightningModule:
        new_model = deepcopy(model)
        new_model = self._freeze_all_parameters(new_model)

        self.adapter_method.apply(new_model)

        return new_model

    def _freeze_all_parameters(self, model: LightningModule) -> LightningModule:
        for param in model.parameters():
            param.requires_grad = False

        return model
