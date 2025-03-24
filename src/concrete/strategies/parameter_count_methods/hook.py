from lightning import LightningModule

from src.interfaces.strategies.parameter_count_method import ParameterCountMethod
from src.utils.classification import is_classification_parameter


class Hook(ParameterCountMethod):
    def __init__(self) -> None:
        super().__init__()

    def set_parameter_count(self, model: LightningModule) -> LightningModule:
        param_count = {}
        for name, param in model.named_parameters():
            count = param.numel()
            param_count["total"] = param_count.get("total", 0) + count

            if is_classification_parameter(name, model):
                location = "classification"
            else:
                location = "model"

            count_dict = param_count.get(location, {})

            free_count = getattr(param, "free_count")
            frozen_count = getattr(param, "frozen_count")

            count_dict["free"] = count_dict.get("free", 0) + free_count
            count_dict["frozen"] = count_dict.get("frozen", 0) + frozen_count

            param_count[location] = count_dict

        setattr(model, "parameter_count", param_count)
        return super().set_parameter_count(model)
