from lightning import LightningModule

from src.interfaces.strategies.parameter_count_method import ParameterCountMethod


class Skip(ParameterCountMethod):
    def __init__(self) -> None:
        super().__init__()

    def set_parameter_count(self, model: LightningModule) -> LightningModule:
        setattr(
            model,
            "parameter_count",
            {"total": sum(p.numel() for p in model.parameters())},
        )
        return model
