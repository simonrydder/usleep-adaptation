from copy import deepcopy

from src.config._adapter_config import AdapterMethodConfig
from src.interfaces.adapter import Adapter
from src.interfaces.framework_model import FrameworkModel
from src.utils.classification import is_classification_parameter


class StandardAdapter(Adapter):
    def __init__(self, config: AdapterMethodConfig) -> None:
        super().__init__(config)

        settings = config.settings.get_settings()
        self.adapter_method = config.method(**settings)
        self.param_count_method = config.parameter_count()

    def adapt(self, model: FrameworkModel, **kwargs) -> FrameworkModel:
        new_model = deepcopy(model)
        new_model = self._freeze_all_parameters(new_model)
        new_model = self._unfreeze_segmentation_later(new_model)

        self.adapter_method.apply(new_model, **kwargs)
        new_model = self.param_count_method.set_parameter_count(new_model)

        return new_model

    def _freeze_all_parameters(self, model: FrameworkModel) -> FrameworkModel:
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _unfreeze_segmentation_later(self, model: FrameworkModel) -> FrameworkModel:
        for name, param in model.named_parameters():
            if is_classification_parameter(name, model):
                param.requires_grad = True
                print(f"Unfroze {name}")

        return model
