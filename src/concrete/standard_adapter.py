from copy import deepcopy

from lightning import LightningModule

from src.config._adapter_config import AdapterMethodConfig
from src.interfaces.adapter import Adapter


class StandardAdapter(Adapter):
    def __init__(self, config: AdapterMethodConfig) -> None:
        super().__init__(config)

        settings = config.settings.get_settings()
        self.adapter_method = config.method(**settings)

    def adapt(self, model: LightningModule, **kwargs) -> LightningModule:
        new_model = deepcopy(model)
        new_model = self._freeze_all_parameters(new_model)

        self.adapter_method.apply(new_model, **kwargs)

        return new_model

    def _freeze_all_parameters(self, model: LightningModule) -> LightningModule:
        for param in model.parameters():
            param.requires_grad = False

        return model
