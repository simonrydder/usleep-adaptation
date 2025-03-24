from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.parameter_count_method import ParameterCountMethod
from src.utils.classification import is_classification_parameter


class Grad(ParameterCountMethod):
    def __init__(self) -> None:
        super().__init__()

    def set_parameter_count(self, model: FrameworkModel) -> FrameworkModel:
        param_count = {}

        for name, param in model.named_parameters():
            count = param.numel()
            param_count["total"] = param_count.get("total", 0) + count

            if param.requires_grad:
                type = "free"
            else:
                type = "frozen"

            if is_classification_parameter(name, model):
                location = "classification"
            else:
                location = "model"

            count_dict = param_count.get(location, {})
            count_dict[type] = count_dict.get(type, 0) + count
            param_count[location] = count_dict

        setattr(model, "parameter_count", param_count)
        return model
