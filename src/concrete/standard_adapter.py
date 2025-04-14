from copy import deepcopy

from src.config._adapter_config import AdapterMethodConfig
from src.interfaces.adapter import Adapter
from src.interfaces.framework_model import FrameworkModel


class StandardAdapter(Adapter):
    def __init__(self, config: AdapterMethodConfig) -> None:
        super().__init__(config)

        settings = config.settings.get_settings()
        self.optimizer_settings = config.optimizer_settings.get_settings()
        self.adapter_method = config.method(**settings)
        self.param_count_method = config.parameter_count()

    def adapt(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        new_model = deepcopy(model)
        setattr(new_model, "original_model", False)
        new_model = self._freeze_all_parameters(new_model)
        new_model = self._unfreeze_segmentation_later(new_model)
        self._add_optimizer_settings(new_model)

        self.adapter_method.apply(new_model, **kwargs)
        new_model = self.param_count_method.set_parameter_count(new_model)

        return new_model

    def _freeze_all_parameters(self, model: FrameworkModel) -> FrameworkModel:
        for param in model.parameters():
            param.requires_grad = False

        print("All parameters are frozen")
        return model

    def _unfreeze_segmentation_later(self, model: FrameworkModel) -> FrameworkModel:
        for name, param in model.named_parameters():
            if model.is_classification_parameter(name):
                param.requires_grad = True

        print("Segment classifier parameters are unfrozen")

        return model

    def _add_optimizer_settings(self, new_model: FrameworkModel) -> None:
        setattr(new_model, "lr", self.optimizer_settings.get("lr"))
        setattr(new_model, "lr_patience", self.optimizer_settings.get("lr_patience"))
        setattr(new_model, "lr_minimum", self.optimizer_settings.get("lr_minimum"))
        setattr(new_model, "lr_factor", self.optimizer_settings.get("lr_factor"))
