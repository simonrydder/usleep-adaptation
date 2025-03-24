from src.interfaces.framework_model import FrameworkModel
from src.interfaces.strategies.parameter_count_method import ParameterCountMethod


class Skip(ParameterCountMethod):
    def __init__(self) -> None:
        super().__init__()

    def set_parameter_count(self, model: FrameworkModel) -> FrameworkModel:
        setattr(
            model,
            "parameter_count",
            {"total": sum(p.numel() for p in model.parameters())},
        )
        return model
